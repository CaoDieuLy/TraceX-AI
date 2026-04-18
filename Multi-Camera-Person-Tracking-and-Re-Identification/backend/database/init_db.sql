-- =====================================================
-- SCHEMA: Video Queue Management System
-- =====================================================

-- 1. Videos table: lưu metadata cho mỗi video file
CREATE TABLE IF NOT EXISTS videos (
    id SERIAL PRIMARY KEY,
    filename VARCHAR(500) UNIQUE NOT NULL,
    drive_link_h265 TEXT NOT NULL,
    drive_link_metadata TEXT NOT NULL,
    drive_file_id_h265 VARCHAR(200),
    drive_file_id_metadata VARCHAR(200),
    status VARCHAR(50) DEFAULT 'available' CHECK (status IN ('available', 'processing', 'deleted', 'archived')),
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP,
    file_size_bytes BIGINT,
    duration_seconds FLOAT,
    resolution VARCHAR(50),
    camera_id VARCHAR(100),
    notes TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Queue log: lịch sử xử lý queue (FIFO tracking)
CREATE TABLE IF NOT EXISTS queue_log (
    id SERIAL PRIMARY KEY,
    video_id INTEGER REFERENCES videos(id) ON DELETE CASCADE,
    action VARCHAR(50) NOT NULL CHECK (action IN ('enqueue', 'dequeue', 'delete', 'archive')),
    action_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    reason TEXT,
    metadata JSONB
);

-- 3. Indexes cho performance
CREATE INDEX idx_videos_status ON videos(status);
CREATE INDEX idx_videos_uploaded_at ON videos(uploaded_at DESC);
CREATE INDEX idx_videos_camera_id ON videos(camera_id);
CREATE INDEX idx_queue_log_video_id ON queue_log(video_id);
CREATE INDEX idx_queue_log_action_at ON queue_log(action_at DESC);

-- 4. Trigger tự động cập nhật updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_videos_updated_at
    BEFORE UPDATE ON videos
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- 5. Function: Lấy video available tiếp theo (FIFO - oldest first)
CREATE OR REPLACE FUNCTION get_next_video_for_processing()
RETURNS TABLE (
    video_id INTEGER,
    filename VARCHAR,
    drive_link_h265 TEXT,
    drive_link_metadata TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT v.id, v.filename, v.drive_link_h265, v.drive_link_metadata
    FROM videos v
    WHERE v.status = 'available'
    ORDER BY v.uploaded_at ASC  -- FIFO: lấy cũ nhất
    LIMIT 1
    FOR UPDATE SKIP LOCKED;  -- Tránh race condition nếu có nhiều worker
END;
$$ LANGUAGE plpgsql;

-- 6. Function: Đếm số video available
CREATE OR REPLACE FUNCTION count_available_videos()
RETURNS INTEGER AS $$
BEGIN
    RETURN (SELECT COUNT(*) FROM videos WHERE status = 'available');
END;
$$ LANGUAGE plpgsql;

-- 7. Function: cleanup old videos (FIFO - giữ tối đa N video mới)
CREATE OR REPLACE FUNCTION cleanup_old_videos(max_keep INTEGER DEFAULT 100)
RETURNS INTEGER AS $  -- Trả về số record bị xóa
$$
DECLARE
    deleted_count INTEGER;
BEGIN
    WITH to_delete AS (
        SELECT id FROM videos
        WHERE status = 'available'
        ORDER BY uploaded_at DESC
        OFFSET max_keep
    )
    DELETE FROM videos v
    USING to_delete td
    WHERE v.id = td.id
    RETURNING COUNT(*) INTO deleted_count;

    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- 8. View: trạng thái queue hiện tại
CREATE OR REPLACE VIEW v_queue_status AS
SELECT
    status,
    COUNT(*) as count,
    MIN(uploaded_at) as oldest_uploaded,
    MAX(uploaded_at) as newest_uploaded
FROM videos
GROUP BY status;

-- =====================================================
-- SAMPLE DATA (temporary for testing)
-- =====================================================
-- INSERT INTO videos (filename, drive_link_h265, drive_link_metadata)
-- VALUES
--     ('camera01_20260418T080000Z.h265', 'https://drive.google.com/...', 'https://drive.google.com/...'),
--     ('camera02_20260418T080500Z.h265', 'https://drive.google.com/...', 'https://drive.google.com/...');

-- =====================================================
-- USAGE EXAMPLES
-- =====================================================
-- Lấy video tiếp theo để xử lý:
-- SELECT * FROM get_next_video_for_processing();

-- Đếm số video available:
-- SELECT count_available_videos();

-- Cleanup giữ tối đa 100 video:
-- SELECT cleanup_old_videos(100);

-- Xem trạng thái queue:
-- SELECT * FROM v_queue_status;

COMMIT;

# Bao cao ky thuat chi tiet - model pipeline tren toan bo workflow

Ngay cap nhat: 2026-04-28

## 1. Muc tieu cua bao cao

Tai lieu nay mo ta toan bo workflow ky thuat cua he thong theo 3 truc:

- storage pipeline: du lieu video di tu dau vao den queue, artifact, DB nhu the nao;
- model pipeline: moi stage dang dung model gi, output gi, va stage sau tieu thu gi;
- coding flow: module nao trong code hien tai chiu trach nhiem orchestration, module nao la runtime, module nao moi la production target.

Bao cao nay co y do rat ro:

- khong mo ta kien truc theo kieu "service nao cung lam duoc";
- khong hop thuc hoa fallback cu;
- phan biet ro `contract hien tai`, `local runnable path`, va `production strict runtime`.
- chi ghi ten cac model da ton tai trong repo hoac da duoc chot trong `strict_pipeline.py`;
- model nao chua co trong repo thi khong duoc viet nhu da implementation xong; neu can dung, no phai di qua chuoi `download -> fine-tune -> adapter hoa -> deploy`.

## 2. Tong quan workflow muc tieu

Workflow muc tieu cua he thong la:

```text
temp/*.mp4
-> move.py
-> storage/cam_xx/yyyy-mm-dd/*.mp4
-> storage_ingest scanner
-> post_move_ingestion contract
-> metadata queue runtime
-> ai_service
-> tracking-service / LightningAI
-> decode + sample 5 fps
-> detection
-> tracking
-> tracklet quality scoring
-> tracklet feature pipeline
   - attribute
   - appearance
   - action
-> metadata artifact
-> queue/db/search index
```

Workflow nay co 3 lop can tach ro:

1. `storage + queue orchestration`
2. `tracking / feature runtime`
3. `metadata persistence + search`

Neu 3 lop nay bi tron vao nhau, code se mat tinh on dinh va rat de quay lai tinh trang:

- local path va production path tron lan;
- runtime contract va fallback contract song song;
- metadata cu va metadata moi cung ton tai.

## 3. Kien truc active theo repo hien tai

He thong duoc tach thanh 4 khoi chinh:

- `frontend/`: giao dien va workflow nguoi dung
- `backend/`: metadata-service + public gateway
- `ai_service/`: bien gioi AI noi bo
- `tracking-service/`: ingestion runtime, tracklet runtime, feature runtime

Phan cong active:

- `move.py`: chi lam storage normalization
- `backend/services/metadata-service/app/storage_ingest.py`: scan video moi trong `storage/`
- `backend/services/metadata-service/app/post_move_ingestion.py`: dinh nghia contract ingestion sau `move.py`
- `backend/services/metadata-service/app/queue_runtime.py`: orchestration queue va goi upstream
- `ai_service/aiapp/main.py`: proxy noi bo toi tracking runtime
- `backend/services/tracking-service/app/strict_pipeline.py`: khai bao strict production pipeline
- `backend/services/tracking-service/app/local_ingestion_pipeline.py`: local runnable substitute path
- `backend/services/tracking-service/app/tracklet_feature_pipeline.py`: contract va xu ly feature sau tracklet quality scoring

### 3.1 Nguon du lieu muc tieu: PhysicalAI-SmartSpaces tren Google Drive

Nguon du lieu muc tieu cua he thong la `PhysicalAI-SmartSpaces`, nhung trong van hanh thuc te no se nam tren Google Drive cua he thong va di vao pipeline qua storage root, khong duoc mac dinh local path development la source of truth.

He thong can duoc hieu theo luong:

```text
Google Drive
-> temp/
-> move.py
-> storage/
-> queue
-> LightningAI
```

Theo dataset card cua `nvidia/PhysicalAI-SmartSpaces`, bo du lieu nay co mot so tinh chat quyet dinh truc tiep den thiet ke:

- hon `250 gio` video dong bo;
- gan `1,500 camera`;
- `MTMC_Tracking_2024` co dung luong khoang `216.95 GB`;
- video chuan la `MP4 (H.264)`, `1080p`, `30 FPS`;
- nhan MTMC gom `camera_id`, `obj_id`, `frame_id`, bbox, va toa do the gioi.

Y nghia ky thuat:

- detector/tracker phai du suc cho bai toan da camera, indoor, occlusion cao;
- storage pipeline phai toi uu cho volume lon va retention dai;
- sampling `5 fps` la quyet dinh bat buoc de tranh no compute;
- metadata phai giu `tracklet-level identity + geometry + semantic context` de phuc vu MTMC search.

## 4. Storage pipeline va yeu cau luu tru cot loi

### 4.1 Drive root

Drive root duoc to chuc theo 2 nhanh:

```text
temp/
storage/
```

`temp/` la noi file moi xuat hien:

```text
temp/
  cam_01_2026-04-19_12-10.mp4
  cam_02_2026-04-19_12-10.mp4
  ...
```

`storage/` la source-of-truth sau normalization:

```text
storage/
  cam_01/
    2026-04-19/
      cam_01_2026-04-19_10-00.mp4
      cam_01_2026-04-19_10-10.mp4
      ...
  cam_02/
    2026-04-19/
      cam_02_2026-04-19_10-00.mp4
      ...
```

### 4.2 Vai tro cua `move.py`

`move.py` khong phai model stage. No la storage stage.

Input:

- file trong `temp/`
- ten file theo pattern `cam_xx_yyyy-mm-dd_hh-mm.mp4`

Output:

- file duoc move vao `storage/cam_xx/yyyy-mm-dd/`
- ten file duoc giu nguyen
- container van la `.mp4`

Y nghia ky thuat:

- storage scanner chi can scan 1 pattern duy nhat;
- queue khong can hieu logic temp;
- khong can detect `.h265` theo suffix nua.

### 4.3 Dac tinh du lieu dau vao

Video dau vao hien tai co 3 dac diem cot loi:

1. file duoc luu duoi duoi `.mp4`
2. payload ben trong la `H.265`
3. fps goc khong on dinh, co the 15 fps, 30 fps, hoac do ghep nhieu nguon

Dieu nay rat quan trong vi no dan den quyet dinh:

```text
decode -> sample 5 fps
```

He thong khong duoc de detector/tracker xu ly theo fps goc, vi khi do chi phi compute no rat nhanh.

### 4.4 Do dai, so luong file, va retention

Neu scope van hanh la:

- `50 cameras`
- ingest moi `10 phut`
- retention `30 ngay`

Thi so luong file la:

- `6 file/gio/camera`
- `144 file/ngay/camera`
- `7200 file/ngay/50 cameras`
- `216000 file/30 ngay`

Neu moi clip la `10 phut` va sample ve `5 fps`:

- `3000 sampled frames/video`
- `150000 sampled frames/50 videos trong mot batch 10 phut`

Day la ly do storage pipeline phai:

- chia folder theo `camera/date`
- scanner phai idempotent
- queue phai xu ly theo batch va job window
- metadata phai theo `tracklet-level`, khong theo `frame-level`

## 5. Contract ingestion sau `move.py`

File chiu trach nhiem la:

`backend/services/metadata-service/app/post_move_ingestion.py`

No dinh nghia `PostMoveIngestionPolicy` gom 4 phan:

1. `DecodeSamplingPolicy`
2. `TrackingPolicy`
3. `TrackletQualityPolicy`
4. `TrackletFeaturePipelinePolicy`

### 5.1 DecodeSamplingPolicy

Contract active hien tai:

- `container_suffix = ".mp4"`
- `codec = "h265"`
- `sample_fps = 5`
- `keep_container_as_mp4 = True`

Y nghia:

- he thong chot ro container va codec
- queue/runtime khong can tranh cai lai suffix
- detector/tracker nhan input da duoc sample theo mot contract co dinh

### 5.2 TrackingPolicy

Contract active:

- detector: `RF-DETR 2x-large`
- official class: `RFDETR2XLarge`
- inference alias: `rfdetr-2xlarge`
- package: `rfdetr[plus]`
- tracker: `OCMCTrack-style corrective cascade`
- scope: `per_video`
- rule: `each_10_min_video_is_independent`
- output fields: `video_id`, `track_id`, `bboxes`

Y nghia:

- tracklet duoc dinh nghia theo tung video 10 phut
- khong track xuyen video tai stage nay
- output trung tam la danh sach bbox theo frame trong tracklet

### 5.3 TrackletQualityPolicy

Contract active:

- `minimum_confidence_score = 0.3`
- `minimum_tracklet_duration_seconds = 2.0`
- reject blurry tracklets
- reject incomplete tracklets

Y nghia:

- quality scoring khong chi la classifier filter
- no la bo loc tai nguyen cho metadata pipeline phia sau
- neu bo loc nay qua long, action branch va DB se no nhanh

### 5.4 TrackletFeaturePipelinePolicy

Contract active:

- `selection_mode = hybrid_track_mean_plus_topk`
- `keyframe_selector = laplacian_variance_with_bbox_area_rank`
- score inputs:
  - `laplacian_variance`
  - `bbox_area`
  - `detection_confidence`
- `selected_frame_count = 5`
- `max_selected_frame_count = 10`
- pooling methods:
  - `mean_pooling`
  - `max_pooling`
  - `quality_weighted_mean`

Metadata fields sau do:

- attribute:
  - `gender`
  - `age_group`
  - `attribute_embedding_vector`
- appearance:
  - `head_accessory`
  - `hat`
  - `hair_color`
  - `skin_tone`
  - `shirt`
  - `pants`
  - `shoes`
  - `bag`
  - `appearance_embedding_vector`
  - `embedding_vector`
  - `tracklet_vectors`
- action:
  - `timeline`
  - `matched_segments`
  - `action_semantic_embedding`

Action clip parameters:

- `action_clip_duration_seconds = 2.0`
- `action_clip_stride_seconds = 1.5`
- semantic model target: `ITSELF`

## 6. Strict production pipeline

File chiu trach nhiem:

`backend/services/tracking-service/app/strict_pipeline.py`

Day la noi repo khai bao pipeline production muc tieu, khong phai local substitute path.

### 6.1 Detector

Strict production detector:

- ten: `RF-DETR 2x-large`
- official class: `RFDETR2XLarge`
- family: `detection_transformer`
- `nms_free = True`
- `predict_api = model.predict(image, threshold=...)`
- deploy target: `TensorRT FP16`
- resolution: `880x880`

Detector hyperparameters:

- `confidence_threshold = 0.32`
- `person_class_only = True`
- `max_queries = 300`

Tac dung trong workflow:

- nhan frame da sample 5 fps
- tra bounding boxes cho class nguoi
- day sang tracker

### 6.2 Tracker

Strict production tracker:

- ten: `OCMCTrack-style corrective cascade`
- family: `online_mtmc`
- co `geometry_aware`
- co `occlusion_reasoning`

Tracker hyperparameters active:

- `high_confidence_threshold = 0.45`
- `low_confidence_threshold = 0.12`
- `new_track_threshold = 0.55`
- `iou_gate = 0.18`
- `appearance_gate = 0.22`
- `corrective_buffer_seconds = 14.0`
- `world_gate_max_speed_mps = 2.8`

Tac dung trong workflow:

- nhan detections cua tung video
- gan detections thanh tracklet online
- sua association bang geometry + appearance + corrective buffer

### 6.3 Appearance embedding / ReID

Strict production appearance embedding:

- ten: `SOLIDER + KPR`
- family: `human_foundation_reid`
- `part_based = True`
- `occlusion_robust = True`

ReID hyperparameters active:

- `global_embedding_weight = 0.55`
- `part_embedding_weight = 0.45`
- `cross_camera_match_threshold = 0.27`

Vai tro:

- sinh vector appearance cho tracklet
- ho tro matching va search ranking
- giu duoc thong tin occlusion tot hon backbone nhe kieu baseline

### 6.4 Semantic retrieval / action-text alignment

Strict production semantic stage:

- ten: `ITSELF`
- family: `vision_language_retrieval`
- `fine_grained_alignment = True`

Semantic search weights:

- `embedding = 0.58`
- `semantic_overlap = 0.22`
- `visibility = 0.12`
- `world_position = 0.08`

Vai tro:

- khong chi tra text-match thong thuong
- phuc vu truy van chi tiet theo mo ta hanh vi / ngoai hinh
- la stage semantic o lop search, khong phai detector hay tracker

### 6.5 Evaluation

Strict production evaluation:

- `TrackEval HOTA`

Y nghia:

- danh gia tong hop detection-association tracking quality
- phu hop hon viec chi do MOTA hoac chi do ReID re roi

## 7. Local runnable path trong code hien tai

File chiu trach nhiem:

`backend/services/tracking-service/app/local_ingestion_pipeline.py`

Can noi thang:

- day khong phai strict production runtime
- day la local runnable path de repo co the chay end-to-end o muc logic

### 7.1 VideoFrameSampler

Class:

- `VideoFrameSampler`

Vai tro:

- mo video bang OpenCV
- doc fps goc
- sample lai ve `5 fps`
- tinh `laplacian_score` tung frame

Output:

- `SampledFrame`
  - `frame_index`
  - `timestamp_second`
  - `image`
  - `laplacian_score`

### 7.2 Local detector

Class:

- `HogPersonDetector`

Canh bao ky thuat:

- day la lightweight detector de chay local
- khong phai RF-DETR production

Vai tro:

- phat hien nguoi tren frame sample
- fallback contour detection khi HOG khong ra detection

### 7.3 Local tracker

Class:

- `GreedyIoUTracker`

Canh bao ky thuat:

- day la tracker local de repo chay duoc
- khong phai OCMCTrack-style corrective cascade production

Vai tro:

- gan detections theo IoU qua tung frame
- tao `LocalTracklet`

### 7.4 TrackletQualityScorer

Class:

- `TrackletQualityScorer`

Nguong local active:

- `minimum_confidence_score = 0.3`
- `minimum_duration_seconds = 2.0`
- `minimum_frame_count = 3`
- `minimum_average_laplacian = 12.0`

Vai tro:

- accept/reject tracklet local
- day vao feature pipeline

## 8. Tracklet feature pipeline trong code hien tai

File chiu trach nhiem:

`backend/services/tracking-service/app/tracklet_feature_pipeline.py`

Day la phan quan trong nhat cua bao cao nay, vi day la noi gop ca `attribute`, `appearance`, `action`.

### 8.1 Input contract

Input trung tam cua feature pipeline la:

- `TrackletFeatureInput`
  - `video_id`
  - `object_id`
  - `sampled_fps`
  - `frames`

Moi frame trong tracklet la:

- `TrackletFrameObservation`
  - `frame_index`
  - `timestamp_second`
  - `bbox`
  - `detection_confidence`
  - `laplacian_score`

Y nghia:

- don vi xu ly khong phai frame re
- don vi xu ly la `tracklet`

### 8.2 Frame selection

Class lien quan:

- `FrameSelectionWeights`
- `FrameQualityScore`
- `FrameSelectionOutput`

Pipeline dang co 2 tin hieu song song:

1. `average_tracklet_score`
2. `representative_frame`

Dong thoi no con giu:

- `selected_frames`
- `ranked_frames`
- `pooling_scores`

Logic nay phu hop voi yeu cau nghiep vu:

- co huong lay trung binh tren toan bo tracklet
- co huong lay top `5-10` frame tot nhat

Y nghia ky thuat:

- metadata khong bi le thuoc 1 frame duy nhat
- van co frame dai dien de preview va debug
- pooling co the nang cap dan tu heuristic sang learned attention

### 8.3 Attribute stage

Dataclass output:

- `StaticAttributeResult`
  - `gender`
  - `age_group`
  - `confidence`

- `AttributeEmbeddingResult`
  - `embedding_model`
  - `embedding_vector`

Hien trang code:

- `RuleBasedStaticAttributeExtractor`
- `VisualAttributeEmbeddingExtractor`

Danh gia:

- contract IO da ro
- local path da khong con null placeholder o stage nay
- hien tai van la extractor xap xi/rule-based, chua phai model production fine-tuned
- pipeline dang cho san cho adapter production

### 8.4 Appearance stage

Dataclass output:

- `AppearanceAttributeResult`
  - `head_accessory`
  - `hat`
  - `hair_color`
  - `skin_tone`
  - `shirt`
  - `pants`
  - `shoes`
  - `bag`

Appearance embedding outputs:

- `appearance_embedding_vector`
- `embedding_vector`
- `tracklet_vectors`

Hien trang code:

- `ColorAppearanceAttributeExtractor`
- `VisualAppearanceEmbeddingExtractor`

Danh gia:

- schema metadata da ro
- vector contract da ro
- local path da co crop-aware metadata extractor va quality-weighted visual embedding
- model production cho metadata va embedding van chua noi day du trong local path

### 8.5 Action stage

Dataclass output:

- `ActionClip`
- `BehaviorAnalysisResult`
- `SemanticEmbeddingResult`

Hien trang code:

- `HeuristicBehaviorAnalyzer`
- `ActionVocabularyEmbedder`

Y nghia:

- action stage da noi vao pipeline
- output da co:
  - `timeline`
  - `matched_segments`
  - `action_semantic_embedding`
- nhung model that cho action recognition / VLM chua duoc thay the trong local path

### 8.6 Aggregation stage

Dataclass ket qua tong hop:

- `TrackletFeatureAggregationOutput`

Output cap tracklet:

- `attribute_summary`
- `appearance_summary`
- `semantic_attributes`
- `attribute_embedding_vector`
- `appearance_embedding_vector`
- `embedding_vector`
- `tracklet_vectors`
- `bbox_samples`
- `timeline`
- `matched_segments`
- `action_semantic_embedding`

Y nghia:

- ca `attribute`, `appearance`, `action` cung tro thanh metadata cua cung mot tracklet
- day la output hop ly de dua vao DB va search

## 9. Model pipeline ly tuong va model pipeline hien tai

### 9.1 Production target

Neu theo strict runtime, model pipeline ly tuong la:

```text
video .mp4(H.265)
-> decode/sample 5 fps
-> RF-DETR 2x-large
-> OCMCTrack-style corrective cascade
-> tracklet quality scoring
-> frame selection
-> attribute model that
-> appearance metadata model that
-> SOLIDER + KPR
-> action recognition / VLM that
-> ITSELF semantic stage
-> metadata + vectors + ranking
```

### 9.2 Local code path hien tai

Local code path hien tai thuc te la:

```text
video .mp4(H.265)
-> OpenCV sampler 5 fps
-> HOG detector
-> Greedy IoU tracker
-> local tracklet quality scoring
-> frame selection
-> rule-based static attribute extractor
-> crop-aware appearance extractor
-> visual embeddings fused tren selected frames
-> heuristic action analyzer
-> heuristic action semantic embedding
-> metadata artifact
```

Canh bao:

- local path dung de chay logic end-to-end
- khong duoc danh dong voi production strict runtime

### 9.3 Ket luan quan trong

Code hien tai da co:

- workflow storage ro rang
- ingestion contract ro rang
- tracklet feature pipeline ro rang
- metadata output ro rang

Nhung code hien tai chua co day du:

- RF-DETR runtime that trong local path
- OCMCTrack-style runtime that trong local path
- attribute model production that
- appearance metadata model production that
- action model/VLM that

Noi cach khac:

- orchestration da ro
- model contracts da ro
- production model stack da duoc chot
- local runtime da giam placeholder, nhung van con stage rule-based/heuristic o attribute va action
- model nao chua co trong repo hien tai van chi duoc xem la muc tieu can bo sung, chua duoc xem la active implementation

## 10. Song song hoa va dat batching o dau

He thong nay can xu ly batch lon, nhung khong phai moi lop deu duoc phep batch.

Nguyen tac uu tien cua he thong la:

1. `accuracy truoc`, sau do moi toi uu throughput;
2. throughput phai den tu batching va parallelism dung lop runtime, khong duoc den tu viec giam model mot cach vo ky luat;
3. detector, tracker, ReID, va action stage deu phai duoc dat trong GPU runtime that, tranh bottleneck o queue layer.

### 10.1 Lop duoc phep song song hoa

- queue job level
- video ingestion level
- detector batch level
- crop inference batch level
- metadata worker level

### 10.2 Lop khong nen nhan batching model

`queue_runtime.py` khong nen tro thanh noi quan ly tensor, crop, hay VRAM.

Ly do:

- day la orchestration layer
- no nen biet video task, queue task, DB task
- no khong nen biet chi tiet GPU memory plan

### 10.3 Noi dat batching hop ly

Batching model that nen nam o:

- `tracking-service`
- hoac LightningAI runtime

Vi day moi la noi nam:

- selected frames
- crop tensors
- detector batches
- ReID batches
- action clip batches
- execution plan theo hardware detect duoc

Tu goc nhin van hanh, bai toan nay uu tien cach dung tai nguyen nhu sau:

- giam frame volume bang `5 fps`;
- giam crop volume bang `tracklet quality scoring` va `selected_frame_count = 5`, `max_selected_frame_count = 10`;
- dua detector batches, ReID crop batches, action clip batches vao LightningAI runtime;
- de metadata-service chi lam queue orchestration, retry, va persistence;
- de tracking-service/LightningAI tu quyet batch size va stream scheduling dua tren GPU thuc te.

## 11. Pain points chinh cua workflow hien tai

### 11.1 Contract va runtime chua gap nhau hoan toan

Repo da chot strict pipeline:

- RF-DETR
- OCMCTrack-style
- SOLIDER + KPR
- ITSELF

Nhung local runnable path van dung:

- HOG
- Greedy IoU
- heuristic extractors

Do do, repo hien dang manh ve architecture va weak hon o production runtime completeness.

### 11.2 Lightning runtime co the chay code cu

Trong van hanh that, da xuat hien blocker:

- repo local da doi sang `.mp4`
- nhung Lightning runtime ben ngoai van co path cu chi nhan `.h265/.hevc`

Y nghia:

- local repo sach chua du
- production endpoint cung phai dong bo code va contract

### 11.3 Metadata cu va metadata moi rat de song song neu khong don sach

Neu de lai:

- legacy import
- field fallback cu
- duplicate metadata routes
- duplicate secret/infra trees

thi rebase va deploy se quay lai loi cu rat nhanh.

## 12. Dinh huong giai quyet dung

### 12.1 O lop storage

- giu duy nhat luong `temp -> move.py -> storage`
- scanner chi quet `storage/cam_xx/yyyy-mm-dd/*.mp4`
- khong quay lai detect suffix `.h265`

### 12.2 O lop runtime

- coi `strict_pipeline.py` la production source-of-truth
- local runnable path chi dung de smoke test logic
- model that phai duoc noi vao adapter trong `tracklet_feature_pipeline.py`

### 12.3 O lop model

Can thay placeholder bang model that theo 3 nhom:

1. attribute metadata + attribute embedding
2. appearance metadata + appearance embedding
3. action recognition / VLM + action embedding

Dong thoi:

- giu `SOLIDER + KPR` cho appearance embedding neu van bam strict pipeline
- khong dat batching model o `queue_runtime.py`
- dat batching o tracking runtime / LightningAI

Trong pha hien tai, chi nen xem cac thanh phan sau la da duoc chot bang code/contract trong repo:

- `RF-DETR 2x-large`
- `OCMCTrack-style corrective cascade`
- `SOLIDER + KPR`
- `ITSELF`

Attribute model that, appearance metadata model that, va action model/VLM that van thuoc nhom can bo sung implementation va fine-tune theo taxonomy cua he thong.

### 12.4 O lop persistence

DB va metadata artifact chi nen luu output cap tracklet:

- identity context
- geometry context
- quality context
- attribute metadata/vector
- appearance metadata/vector
- action metadata/vector

Khong nen luu lai qua nhieu frame-level intermediate neu khong phuc vu debug.

## 13. Ket luan

Workflow ky thuat cua repo hien tai da ro hon rat nhieu so voi giai doan tron luong truoc day, vi da co:

- storage root ro rang
- `move.py` ro vai tro
- ingestion contract sau storage ro rang
- strict production pipeline duoc chot thanh van ban code
- feature pipeline cap tracklet duoc typed ro input/output

Nhung can nhin dung ban chat:

- phan `storage -> queue -> contract -> metadata schema` da kha on
- phan `model production that` van chua duoc noi day du vao local runnable path
- muon end-to-end production that, runtime tren LightningAI/VPS phai chay cung contract va cung code version
- PhysicalAI-SmartSpaces la du lieu muc tieu de toi uu accuracy, nhung van phai di qua Google Drive va storage pipeline da chot
- uu tien accuracy phai di cung ky luat batching va song song hoa dung lop runtime

Noi gon:

- code hien tai da co bo xuong song cho model pipeline
- strict production stack da duoc chot
- viec con lai la dong bo runtime production va thay the placeholder bang model that ma khong pha contract hien tai

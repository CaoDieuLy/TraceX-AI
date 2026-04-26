from __future__ import annotations


class VLM_Metadata_Engine:
    def __init__(self, use_mock: bool = False) -> None:
        raise RuntimeError(
            "Local legacy metadata captioning has been removed. "
            "Use the strict external AI runtime configured through tracking-service."
        )

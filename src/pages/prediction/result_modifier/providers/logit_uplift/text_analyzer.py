from __future__ import annotations

import hanlp

from src.utils.logger import setup_logger

logger = setup_logger("page3", "prediction")

_hanlp_pipeline = None

USE_HANLP_SEGMENTATION = True
HANLP_MIN_WORD_LENGTH = 2
HANLP_USE_NER = True
HANLP_NER_TYPES = ["ORGANIZATION", "PERSON", "LOCATION"]


def _init_hanlp_pipeline():
    global _hanlp_pipeline
    if _hanlp_pipeline is None:
        try:
            logger.debug("初始化 HanLP pipeline...")
            _hanlp_pipeline = hanlp.load(
                hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH
            )
            logger.debug("HanLP pipeline 初始化完成")
        except Exception as e:
            logger.error(f"HanLP pipeline 初始化失败: {str(e)}", exc_info=True)
            raise
    return _hanlp_pipeline


def hanlp_analyzer(text: str) -> list[str]:
    if not USE_HANLP_SEGMENTATION:
        return []

    if not isinstance(text, str) or not text.strip():
        return []

    try:
        pipeline = _init_hanlp_pipeline()
        result = pipeline(text)

        tokens = []
        entities = []

        if "tok" in result:
            tokens = result["tok"]

        if HANLP_USE_NER and "ner" in result:
            ner_result = result["ner"]
            if isinstance(ner_result, list):
                for entity in ner_result:
                    entity_type = ""
                    entity_text = ""
                    if isinstance(entity, tuple) and len(entity) >= 3:
                        entity_text = entity[0]
                        entity_type = entity[2] if len(entity) > 2 else ""
                    elif isinstance(entity, dict):
                        entity_text = entity.get("text", "")
                        entity_type = entity.get("type", "")

                    if entity_text and (not HANLP_NER_TYPES or entity_type in HANLP_NER_TYPES):
                        entities.append(entity_text)

        words = set()

        for token in tokens:
            if isinstance(token, str):
                token = token.strip()
                if len(token) >= HANLP_MIN_WORD_LENGTH:
                    words.add(token)
            elif isinstance(token, list):
                for t in token:
                    t_str = str(t).strip()
                    if len(t_str) >= HANLP_MIN_WORD_LENGTH:
                        words.add(t_str)

        for entity in entities:
            entity_str = str(entity).strip()
            if len(entity_str) >= HANLP_MIN_WORD_LENGTH:
                words.add(entity_str)

        return sorted(list(words))

    except Exception as e:
        logger.warning(f"HanLP 分析文本失败: {str(e)}, 文本: {text[:50]}...")
        return []

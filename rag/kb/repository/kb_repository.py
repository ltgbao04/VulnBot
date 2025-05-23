from rag.kb.models.kb_document_model import KnowledgeBaseModel, KnowledgeBaseSchema
from utils.session import with_session
from utils.log_common import build_logger

logger = build_logger()

@with_session
def add_kb_to_db(session, kb_name, kb_info, vs_type, embed_model):
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )
    if not kb:
        kb = KnowledgeBaseModel(
            kb_name=kb_name, kb_info=kb_info, vs_type=vs_type, embed_model=embed_model
        )
        session.add(kb)
    else:  # update kb with new vs_type and embed_model
        kb.kb_info = kb_info
        kb.vs_type = vs_type
        kb.embed_model = embed_model
    return True


@with_session
def list_kbs_from_db(session, min_file_count: int = -1):
    kbs = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.file_count > min_file_count)
        .all()
    )
    kbs = [KnowledgeBaseSchema.model_validate(kb) for kb in kbs]
    return kbs


@with_session
def kb_exists(session, kb_name):
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )
    status = True if kb else False
    return status


@with_session
def load_kb_from_db(session, kb_name):
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )
    if kb:
        kb_name, vs_type, embed_model = kb.kb_name, kb.vs_type, kb.embed_model
    else:
        kb_name, vs_type, embed_model = None, None, None
    return kb_name, vs_type, embed_model


@with_session
def delete_kb_from_db(session, kb_name):
    kb = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )
    if kb:
        session.delete(kb)
    return True


@with_session
def get_kb_detail(session, kb_name: str) -> dict:
    kb: KnowledgeBaseModel = (
        session.query(KnowledgeBaseModel)
        .filter(KnowledgeBaseModel.kb_name.ilike(kb_name))
        .first()
    )
    if kb:
        return {
            "kb_name": kb.kb_name,
            "kb_info": kb.kb_info,
            "vs_type": kb.vs_type,
            "embed_model": kb.embed_model,
            "file_count": kb.file_count,
            "create_time": kb.create_time,
        }
    else:
        return {}

import json
import os
import time
from typing import Dict, Literal, Tuple, List

import pandas as pd
import streamlit as st

from st_aggrid import AgGrid, JsCode
from st_aggrid.grid_options_builder import GridOptionsBuilder

from config.config import Configs
from rag.kb.base import get_kb_details, get_kb_file_details
from rag.kb.utils.kb_utils import get_file_path, LOADER_DICT
from utils.log_common import build_logger
from web.utils.utils import ApiRequest, check_success_msg, check_error_msg

logger = build_logger()

cell_renderer = JsCode(
    """function(params) {if(params.value==true){return '✓'}else{return '×'}}"""
)

def config_aggrid(
    df: pd.DataFrame,
    columns: Dict[Tuple[str, str], Dict] = {},
    selection_mode: Literal["single", "multiple", "disabled"] = "single",
    use_checkbox: bool = False,
) -> GridOptionsBuilder:
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_column("No", width=40)
    for (col, header), kw in columns.items():
        gb.configure_column(col, header, wrapHeaderText=True, **kw)
    gb.configure_selection(
        selection_mode=selection_mode,
        use_checkbox=use_checkbox,
        pre_selected_rows=st.session_state.get("selected_rows", [0]),
    )
    gb.configure_pagination(
        enabled=True, paginationAutoPageSize=False, paginationPageSize=10
    )
    return gb

def file_exists(kb: str, selected_rows: List) -> Tuple[str, str]:
    """
    Check whether a document file exists in the local knowledge base folder.
    Return the file's name and path if it exists.
    """
    if selected_rows:
        file_name = selected_rows[0]["file_name"]
        file_path = get_file_path(kb, file_name)
        if os.path.isfile(file_path):
            return file_name, file_path
    return "", ""

def knowledge_base_page(api: ApiRequest):
    try:
        kb_list = {x["kb_name"]: x for x in get_kb_details()}
    except Exception as e:
        logger.error(e)
        st.error("Error retrieving knowledge base information. Please check for database connection issues.")
        st.stop()
    kb_names = list(kb_list.keys())

    if (
        "selected_kb_name" in st.session_state
        and st.session_state["selected_kb_name"] in kb_names
    ):
        selected_kb_index = kb_names.index(st.session_state["selected_kb_name"])
    else:
        selected_kb_index = 0

    if "selected_kb_info" not in st.session_state:
        st.session_state["selected_kb_info"] = ""

    def format_selected_kb(kb_name: str) -> str:
        if kb := kb_list.get(kb_name):
            return f"{kb_name} ({kb['vs_type']} @ {kb['embed_model']})"
        else:
            return kb_name

    selected_kb = st.selectbox(
        "Please select or create a new knowledge base:",
        kb_names + ["Create New Knowledge Base"],
        format_func=format_selected_kb,
        index=selected_kb_index,
    )

    if selected_kb == "Create New Knowledge Base":
        with st.form("Create New Knowledge Base"):
            kb_name = st.text_input(
                "New Knowledge Base Name",
                placeholder="Enter a new knowledge base name (Chinese characters not supported)",
                key="kb_name",
            )
            kb_info = st.text_input(
                "Knowledge Base Description",
                placeholder="Brief description for easy agent lookup",
                key="kb_info",
            )

            col0, _ = st.columns([3, 1])
            vs_types = list([Configs.kb_config.default_vs_type])
            vs_type = col0.selectbox(
                "Vector Store Type",
                vs_types,
                index=vs_types.index(Configs.kb_config.default_vs_type),
                key="vs_type",
            )

            col1, _ = st.columns([3, 1])
            with col1:
                embed_models = list([Configs.llm_config.embedding_models])
                index = 0
                embed_model = st.selectbox("Embeddings Model", embed_models, index)

            submit_create_kb = st.form_submit_button("Create")

        if submit_create_kb:
            if not kb_name or not kb_name.strip():
                st.error("Knowledge base name cannot be empty!")
            elif kb_name in kb_list:
                st.error(f"A knowledge base named {kb_name} already exists!")
            elif embed_model is None:
                st.error("Please select an Embeddings Model!")
            else:
                ret = api.create_knowledge_base(
                    knowledge_base_name=kb_name,
                    vector_store_type=vs_type,
                    embed_model=embed_model,
                )
                st.toast(ret.get("msg", " "))
                st.session_state["selected_kb_name"] = kb_name
                st.session_state["selected_kb_info"] = kb_info
                st.rerun()

    elif selected_kb:
        kb = selected_kb
        st.session_state["selected_kb_info"] = kb_list[kb]["kb_info"]
        files = st.file_uploader(
            "Upload Knowledge Files:",
            [i for ls in LOADER_DICT.values() for i in ls],
            accept_multiple_files=True,
        )
        kb_info = st.text_area(
            "Enter Knowledge Base Description:",
            value=st.session_state["selected_kb_info"],
        )

        if kb_info != st.session_state["selected_kb_info"]:
            st.session_state["selected_kb_info"] = kb_info
            api.update_kb_info(kb, kb_info)
        
        st.divider()
        doc_details = pd.DataFrame(get_kb_file_details(kb))
        if not len(doc_details):
            st.info(f"No files available in knowledge base `{kb}`")
        else:
            st.write(f"Files available in knowledge base `{kb}`:")
            doc_details.drop(columns=["kb_name"], inplace=True)
            gb = config_aggrid(doc_details, {}, "multiple")
            doc_grid = AgGrid(doc_details, gb.build(), theme="alpine")

        cols = st.columns(3)
        if cols[1].button("Delete Knowledge Base"):
            ret = api.delete_knowledge_base(kb)
            st.toast(ret.get("msg", " "))
            time.sleep(1)
            st.rerun()

        st.divider()
        st.write("List of documents within files. Double-click to modify. Enter 'Y' in delete column to remove.")


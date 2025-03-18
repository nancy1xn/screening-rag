import streamlit as st
from screening_rag.preprocess.search_qdrant_adverse_media import gen_report
import pandas as pd

def render_markdown(contents, appendices):
    final_contents = ",".join(f"\n - {c}"for c in contents)
    final_appendices = ",".join(f"\n - {a}"for a in (list(map(lambda x:f"[{x[0]} {x[1]}]({x[2]})", appendices))))
    return final_contents, final_appendices

st.write("## Adverse Media Report")
entity_name = st.text_input("Entity Name", help="The keyword used to search CNN news. The format is XXX financial crime")

if st.button("generate report"):
    content, appendix = gen_report(entity_name)
    final_contents, final_appendices = render_markdown(content, appendix)

    st.markdown(
        f"""
        ## Adverse Information Report Headline

        {final_contents}
        """
    )
    st.markdown(
        f"""
        ## Appendix

        {final_appendices}
        """
    )


    # st.write(
    #     pd.DataFrame(gen_report(entity_name))
    # )
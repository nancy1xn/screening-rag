import streamlit as st
from screening_rag.preprocess.search_qdrant_adverse_media import gen_report
import pandas as pd

def render_markdown(contents, appendices):
    contents_split_by_line=list(map(lambda x: f"\n - {x}", contents))
    final_contents= ",".join(contents_split_by_line)
    # final_contents = ",".join(f"\n - {c}"for c in contents)
    
    contents_in_markdown_format= list(map(lambda x:f"[{x[0]} {x[1]}]({x[2]})", appendices))
    contents_in_markdown_format_split_by_line= list(map(lambda x: f"\n - {x}",contents_in_markdown_format))
    final_appendices = ",".join(contents_in_markdown_format_split_by_line)
    # final_appendices = ",".join(f"\n - {a}"for a in (list(map(lambda x:f"[{x[0]} {x[1]}]({x[2]})", appendices))))
    return final_contents, final_appendices

st.write("# Adverse Media Report")
entity_name = st.text_input("Entity Name", help="The keyword used to search CNN news.")

def gen_adverse_media_report(entity_name):

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

        merged_content = f"# Adverse Media Report\n\n## Adverse Information Report Headline\n{final_contents}\n\n## Appendix\n{final_appendices}"

        st.download_button(
        label="Download Markdown",
        data=merged_content,
        file_name="data.md",
        mime="text/markdown",
        icon=":material/download:",
        )


gen_adverse_media_report(entity_name)

    # st.write(
    #     pd.DataFrame(gen_report(entity_name))
    # )
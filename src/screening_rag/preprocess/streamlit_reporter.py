import streamlit as st

from screening_rag.preprocess.cnn_crime_event_searcher import gen_report
from screening_rag.preprocess.cnn_news_searcher import gen_report1


def render_markdown(contents, appendices):
    contents_split_by_line = list(map(lambda x: f"\n - {x}", contents))
    final_contents = ",".join(contents_split_by_line)

    contents_in_markdown_format = list(
        map(lambda x: f"[{x[0]} {x[1]}]({x[2]})", appendices)
    )
    contents_in_markdown_format_split_by_line = list(
        map(lambda x: f"\n - {x}", contents_in_markdown_format)
    )
    final_appendices = ",".join(contents_in_markdown_format_split_by_line)

    return final_contents, final_appendices


st.write("# Adverse Media Report")
entity_name = st.text_input("Entity Name", help="The keyword used to search CNN news.")


def gen_adverse_media_report(entity_name):
    if st.button("generate report"):
        background, appendix1 = gen_report1(entity_name)
        content, appendix = gen_report(entity_name)
        final_background, final_appendices1 = render_markdown(background, appendix1)
        final_contents, final_appendices = render_markdown(content, appendix)

        st.markdown(
            f"""

            ## Background

            {final_background}
            """
        )

        st.markdown(
            f"""

            ## Appendix1

            {final_appendices1}
            """
        )

        st.markdown(
            f"""

            ## Adverse Information Report Headline

            {final_contents}
            """
        )
        st.markdown(
            f"""
            ## Appendix2

            {final_appendices}
            """
        )

        merged_content = f"# Adverse Media Report\n\n## Background\n{final_background}\n\n## Appendix1\n{final_appendices1}\n\n## Adverse Information Report Headline\n{final_contents}\n\n## Appendix2\n{final_appendices}"

        st.download_button(
            label="Download Markdown",
            data=merged_content,
            file_name="data.md",
            mime="text/markdown",
            icon=":material/download:",
        )


if __name__ == "__main__":
    gen_adverse_media_report(entity_name)
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if get_script_run_ctx() is None:
        import sys

        from streamlit.web.cli import main

        sys.argv = ["streamlit", "run", __file__]
        main()

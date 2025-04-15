import streamlit as st

from screening_rag.cnn_crime_searcher_tool import (
    generate_crime_events_report,
)
from screening_rag.cnn_news_searcher_tool import generate_background_report


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


def main(entity_name):
    if st.button("generate report"):
        background, appendix1 = generate_background_report(entity_name)
        content, appendix = generate_crime_events_report(entity_name)
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
    if "__streamlitmagic__" not in locals():
        import streamlit.web.bootstrap

        streamlit.web.bootstrap.run(__file__, False, [], {})
    main(entity_name)

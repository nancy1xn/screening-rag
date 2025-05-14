FROM python:3.9
RUN mkdir APP
COPY dist/screening_rag-0.1.0-py3-none-any.whl /APP
RUN pip install /APP/screening_rag-0.1.0-py3-none-any.whl
CMD ["screening-rag", "init", "--keywords", "JP Morgan Financial Crime,Binance Financial Crime", "--amount", "3"]

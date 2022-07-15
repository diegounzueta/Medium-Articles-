FROM python:3.8.12

# Expose port you want your app on
EXPOSE 8080

# Upgrade pip and install requirements
COPY ./ ./

RUN pip install -U pip
RUN pip install -r requirements.txt


# Run
CMD streamlit run tech_check.py –server.port=8080 –server.address=0.0.0.0 --theme.base "dark"
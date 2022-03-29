from ast import Break
import streamlit as st
from streamlit_player import st_player
import pandas as pd
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib


class app:
    def __init__(self, vid, url):
        self.vid = vid
        self.url = url

        self.pipeline()

    def pipeline(self):

        self.initialise_tool()
        self.video_characteristics()
        self.chapters_doc()
        self.transcript_doc()

    def initialise_tool(self):

        font = {'size': 16}
        matplotlib.rc('font', **font)
        plt.style.use('dark_background')

        with open(self.vid, "r") as f:
            a = json.load(f)

        highlights_dict = a["auto_highlights_result"]
        self.highlights = pd.DataFrame(highlights_dict["results"])
        self.stafety_labels = pd.DataFrame(
            a["content_safety_labels"]["results"])
        self.categories = a["iab_categories_result"]
        self.chapters_dict = a["chapters"]
        self.sentiment_analysis = pd.DataFrame(
            a["sentiment_analysis_results"])
        self.entity_detection = a["entities"]
        self.transcript = a["text"]
        self.words = pd.DataFrame(a["words"])

        st.set_page_config(
            page_title="Podcast Analysis", page_icon="📝", initial_sidebar_state="expanded"
        )

        st.markdown("## SPEACH ANALYSIS TOOL")

        # st.sidebar.markdown(self.transcript)

        self.placeholder = st.empty()
        with self.placeholder.container():
            st_player(self.url, playing=False, muted=True)

    def video_characteristics(self):
        st.markdown("## Video Characteristics")
        st.markdown("Number of Speakers: " +
                    str(len(self.words["speaker"].unique())))
        st.markdown("Top 5 Most Relevant Topics:")
        # st.write(self.categories["summary"])
        keys = list(self.categories["summary"].keys())[:5]
        vals = np.array(list(self.categories["summary"].values())[:5])*100
        fig, ax = plt.subplots(figsize=(10, 4), )
        # fig = plt.figure(facecolor='#0E1117')
        # ax = plt.axes()
        # ax.set_facecolor("#0E1117")
        c = ["C0"]*5
        sns.barplot(x=vals, y=keys, palette=c, ax=ax)
        plt.xlabel("Confidence")
        st.pyplot(fig)

    def chapters_doc(self):
        st.markdown("## Chapters and Summary")
        chapters = pd.DataFrame(self.chapters_dict)

        names_chapters = np.array(["Chapter " + str(i) + ": " +
                                   j for i, j in zip(range(len(chapters)), chapters["gist"])])

        col1, col2 = st.columns((1, 1))
        with col1:
            chap = st.selectbox("Choose your chapter:", names_chapters)
        with col2:

            self.analysis = st.selectbox("Choose analysis: ", [
                "Sentiment", "Highlight", "Safety Warnings"])

        st.write("")

        col1, space, col2 = st.columns((2, 0.1, 1))
        self.selected_chap = chapters[names_chapters == chap].reset_index()
        # st.write(selected_chap)
        with col1:
            st.markdown('<p style="text-align:justify">{}:</p>'.format(
                self.selected_chap["summary"][0]), unsafe_allow_html=True)

        with col2:
            st.markdown("")
            st.markdown("")
            click_chap = st.button("Click to play from chapter")
            url_time = self.url + '&t=' + \
                str(self.selected_chap["start"][0]/1000) + 's'
            if click_chap == True:
                with self.placeholder.container():
                    # st.write(url_time)

                    st_player(url_time, playing=True, muted=False)

    def transcript_doc(self):
        words_chap = self.words.loc[(self.words["start"] >= self.selected_chap["start"][0]) & (
            self.words["end"] <= self.selected_chap["end"][0])].reset_index()
        st.markdown("## Transcript of Chapter:")

        if self.selected_chap["index"][0] == 1:
            for i in range(0, 23):
                words_chap["speaker"][i] = "B"
        sentences = []

        prev_speaker = words_chap["speaker"].values[0]

        sentence = []
        for index, i in words_chap.iterrows():
            if i["speaker"] == prev_speaker:
                sentence.append(i["text"])
            else:

                sentences.append(sentence)
                sentence = []
                sentence.append(i["text"])
                prev_speaker = i["speaker"]
        sentences.append(sentence)

        speaker1, speaker2 = st.columns((1, 1))
        cols = [speaker1, speaker2]

        with speaker1:
            st.markdown('<p style="color: red; text-align:left">JORDAN PETERSON:</p>',
                        unsafe_allow_html=True)
        with speaker2:
            st.markdown('<p style="color: green; text-align:right">JOE ROGAN:</p>',
                        unsafe_allow_html=True)

        # for each sentence (where they are split into speakers)

        for index, i in enumerate(sentences):
            sentence = " ".join(i)

            if self.analysis == "Highlight":
                for j in self.highlights["text"]:
                    if j in sentence:
                        sentence = sentence.replace(
                            j, '<span style="background-color: #6C83A4">' + j + '</span>')

            elif self.analysis == "Sentiment":
                # st.write(self.sentiment_analysis["sentiment"].unique())
                color_dict = {"NEGATIVE": "#B22222",
                              "NEUTRAL": "#6C83A4",
                              "POSITIVE": "#228B22"}

                for j, sentiment in zip(self.sentiment_analysis["text"], self.sentiment_analysis["sentiment"]):
                    color = color_dict[sentiment]
                    if j in sentence:
                        sentence = sentence.replace(
                            j, '<span style="background-color: {color}">'.format(color=color) + j + '</span>')
                        continue

                    elif sentence in j:
                        sentence = '<span style="background-color: {color}">'.format(color=color) + \
                            sentence + '</span>'
                    # else:
                    #     st.write(sentence, j)

            elif self.analysis == "Safety Warnings":
                for _, j in self.stafety_labels.iterrows():

                    if j["text"] in sentence:
                        type = j["labels"][0]
                        for t, v in zip(list(type.keys())[1:], list(type.values())[1:]):
                            type[t] = round(v, 4)
                        sentence = sentence.replace(
                            j["text"], '<span title="{type}"; style="background-color: #B22222">'.format(type=str(type)) + j["text"] + '</span>')

            # st.write(sentence)
            speaker1, space, speaker2 = st.columns((1, 0.2, 1))
            cols = [speaker1, speaker2]
            with cols[index % 2]:
                if index % 2 == 0:
                    st.markdown('<p style="color: white">{}</p>'.format(sentence),
                                unsafe_allow_html=True)
                else:
                    st.markdown('<p style="color: white; text-align:right">{}</p>'.format(sentence),
                                unsafe_allow_html=True)


if __name__ == '__main__':
    app("o4k8b1x6jy-4ca7-45dd-bf5d-8cc3a892ee12.json",
        "https://www.youtube.com/watch?v=YL7rZTV0nHw")

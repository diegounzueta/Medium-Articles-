import pyaudio
import websockets 
import asyncio
import base64
import json 
import streamlit as st
import matplotlib
import matplotlib.pyplot as plt
from api_keys import assemblyAI_key, openaI_key

ass_AI_endpoint = "wss://api.assemblyai.com/v2/realtime/ws?sample_rate=16000"        


class app:
    def __init__(self):
        self.pipeline()
    def pipeline(self):
        self.initialise_app()
    
    def initialise_app(self):
        font = {"size": 16}
        matplotlib.rc("font", **font)
        plt.style.use("dark_background")

        st.set_page_config(
            page_title = "Smart Assistant"
        )

        st.markdown("## SMART ASYSTANT TOOL")




        self.choice = st.radio("", ["RECORD VOICE", "DO NOT RECORD"])

        
        self.statement = st.empty()
        self.respond = st.empty()

        p = pyaudio.PyAudio()
        self.stream = p.open(format = pyaudio.paInt16,
                                channels =1,
                                rate = 16000,
                                input = True,
                                frames_per_buffer = 3200)
        
        asyncio.run(self.send_recieve())






    async def send_recieve(self):

        async with websockets.connect(
            ass_AI_endpoint,
            extra_headers = (("Authentication", assemblyAI_key),),
            ping_interval = 5,
            ping_timeout = 20
        ) as _ws:

            await asyncio.sleep(0.1)
            await _ws.recv()

            async def send():
                while self.choice == "RECORD VOICE":
                    try:
                        data = self.stream.read(3200, exception_on_overflow = False)
                        data = base64.b64encode(data).decode("utf-8")
                        json_data = json.dumps({"audio_data": str(data)})

                        await _ws.send(json_data)


                    except Exception as e:
                        assert False, "Not a websocket 4008 error"
                    await asyncio.sleep(0.01)

                if self.choice != "RECORD VOICE":
                    closeAPI = json.dumps({"terminate_session": True})
                    r = await _ws.send(closeAPI)
                return True

            async def receive():
                while self.choice == "RECORD VOICE":
                    try:
                        result_str = await _ws.recv()
                        if json.load(result_str)["message_type"] == "FinalTranscript":
                            self.statement.empty()
                            with self.statement.container():
                                st.header(json.loads(result_str)["text"])
                    except Exception as e:
                        st.write("ERROR", e)
                        assert False

            send_result, recieve_result = await asyncio.gather(send(), receive())
    # Python 3.7+
    

if __name__ == "__main__":
    app()
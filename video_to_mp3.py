import os
import subprocess
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
files = os.listdir("Videos")
# print(files)

for file in files:
    print(file)
    lecture_number = file.split(" ")[0].split("_")[1]
    topic_name = file.split(" ")[1].split(".")[0]
    # print(f"The number is : {lecture_number}")
    # print(f"The name of the topic is : {topic_name}")
    subprocess.run(['ffmpeg' , '-i' , f"Videos/{file}" , f"audios/{lecture_number} {topic_name}.mp3"])



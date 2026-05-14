# How to use this RAG teaching assistant on your own data : 
## step 1 - Collect your videos
move all your video files to the "videos" folder . 

## step 2 - Covert to mp3
convert all your video files to mp3 by running 'python video_to_mp3.py'.

## step 3 - Convert to text chunks and add to json files . 
convert all your mp3 files to text chunks and add it to json files by running 'python mp3_to_json_chunks.py'.

## step 4 - Convert the json files to vectors : 
convert all your json files to embeddings and store the embeddings into a dataframe by running 'python preprocessing_json.py' and also save it as a joblib pickle . 

## step 5 - Prompt generation and feeding to LLM(deepseek-reasoner) : 
Read the joblib file and load it into the memory . Then create a relevant prompt as per the user query and feed it to the LLM . 






from fastai.vision import open_image, load_learner, image, torch
from fastai import *
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import time
import matplotlib.image as mpimg
import os
import PIL.Image as PIL
import requests
from io import BytesIO
import aiohttp
import asyncio
from pathlib import Path

st.title('Welcome to the new change in Advertising')
imag = Image.open('Banner.jpg')
st.image(imag,use_column_width=True)
my_list=[]

page = st.sidebar.selectbox("Try out the other beta products", ["AI Based Ad Recommender", "FaceAI Feature Extractor"])

if page == "AI Based Ad Recommender":
    
    st.header('Select the features')
    st.subheader('These will required to be entered by the user')
    age = st.multiselect('Specify your age group)', ['Kids (1-15 yrs.)','Young (16-35 yrs.)','Middle Age (35-55 yrs.)','Old (55 yrs. above)'])
    my_list.append(age)
    buying= st.multiselect('Please specify your buying capacity', ['Low Income Group','Low Middle Class','Middle Class','Higher Middle Class','Upper Class'])
    my_list.append(buying)
    profession = st.multiselect('Please specify your profession', ['Students','Office Professionals'])
    my_list.append(profession)


    st.subheader('These will be updated automatically by FaceAI later on')

    gender = st.multiselect('Please specify your gender', ['Male','Female'])
    my_list.append(gender)
    outfit = st.multiselect('Tell us something about your outfit preferences', ['Wearing_Earrings','Wearing_Lipstick','Wearing_Necklace','Wearing_Formals','Wearing_Casuals','Wearing_Necktie'])
    my_list.append(outfit)
    eyewear = st.multiselect('Do you wear any eye wear', ['Specatacles','Sunglasses'])
    my_list.append(eyewear)
    face = st.multiselect('Tell us something about your facial appearance', ['Rosy Cheeks','Chubby Face','Double Chin','Visible Cheek Bones (Fit)','Big Nose','Big Lips','Bushy Eye Brows','Dark Circles','Heavy Makeup'])
    my_list.append(face)
    hair= st.multiselect('What type of hair do you have?', ['Brown Hair','Black Hair','Gray Hair','White Hair','Colour Highlights','Colour Streaks'])
    my_list.append(hair)
    bald= st.multiselect('Are you bald or have curls?', ['Bald','Straight Hair','Wavy Hair','Bangs','Receding Hairline'])
    my_list.append(bald)
    beard= st.multiselect('What kind of beard do you have?', ['No Beard','Slight Beard','Dense Beard','Mustaches','Goatee'])
    my_list.append(beard)


    if st.button('Lets find.'):
        df= pd.DataFrame({'tag_list': [my_list]})
        df.to_csv('Customer_tags.csv')
        cust_tags_list= pd.read_csv('Customer_tags.csv') 
        s = cust_tags_list.tag_list[0]            
    
        def listToString(s):    
            str1 = ""
            for ele in s:  
                str1 += ele     
            return str1 
        hello=listToString(s)
        app=hello.replace('[', '')
        app1=app.replace(']', '')
        app2=app1.replace("'", "")
        app3=app2.replace(" ", "")
        word_set = set(app3.split(',')) 
        mv_tags_list= pd.read_csv('Unedited_advert_nospace.csv', encoding ='latin1')
        mv_tags_list_sim = mv_tags_list[['Title','tag_list','tag_list2','url']] 
        mv_tags_list_sim['jaccard_sim'] = mv_tags_list_sim.tag_list.map(lambda x: len(set(x.split(',')).intersection(set(app3.split(',')))) /len(set(x.split(',')).union(set(app3.split(',')))))
        text = ','.join(mv_tags_list_sim.sort_values(by = 'jaccard_sim', ascending = False).head(25)['tag_list'].values)
        final_list=mv_tags_list_sim.sort_values(by = 'jaccard_sim', ascending = False).head(25)
    
        mv_tags_list_sim['neg_jaccard_sim'] = mv_tags_list_sim.tag_list2.map(lambda x: len(set(x.split(',')).intersection(set(app3.split(',')))) /len(set(x.split(',')).union(set(app3.split(',')))))
        text = ','.join(mv_tags_list_sim.sort_values(by = 'neg_jaccard_sim', ascending = False).head(25)['tag_list2'].values)
        final_list=mv_tags_list_sim.sort_values(['neg_jaccard_sim', 'jaccard_sim'], ascending = [True, False]).head(20)
    
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.005)
            my_bar.progress(percent_complete + 1)
        final_list.to_csv('Recommendations.csv')
        st.header('The Ad Recommendations are:')
        st.dataframe(final_list[['Title']])
        feed= pd.read_csv('Recommendations.csv') 
        i=0
        while i<19:
            url=feed.url[i]
            st.video(url)
     
            i += 1
        
if page == "FaceAI Feature Extractor":
    st.header('Deep Learning backed Facial Feature Extractor')
    st.subheader('Select the mode of image input')
    mode = st.radio('Would you like to run the program on your own face or on someone else face', ['Yes, I would try it on yourself','No, I would like to use an image of someone else on the internet'])
    export_file_url = 'https://www.dropbox.com/s/tv8ianotn9045wc/256_stage-2-rn50.pkl?dl=1'
    export_file_name = '256_stage-2-rn50.pkl'
    path = Path().resolve()
    
    async def download_file(url, dest):
        if dest.exists(): return
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                data = await response.read()
                with open(dest, 'wb') as f:
                    f.write(data)
    
    async def setup_learner():
        await download_file(export_file_url, path / export_file_name)
        try:
            learn = load_learner(path, export_file_name)
            return learn
        except RuntimeError as e:
            if len(e.args) > 0 and 'CPU-only machine' in e.args[0]:
                print(e)
                message = "\n\nThis model was trained with an old version of fastai and will not work in a CPU environment.\n\nPlease update the fastai library in your training environment and export your model again.\n\nSee instructions for 'Returning to work' at https://course.fast.ai."
                raise RuntimeError(message)
            else:
                raise
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tasks = [asyncio.ensure_future(setup_learner())]
    learn = loop.run_until_complete(asyncio.gather(*tasks))[0]
    loop.close()
    
    if mode=='Yes, I would try it on yourself': 
        upload = st.file_uploader("Upload an image", type=("png", "jpg", "jpeg"))
        if st.button('Lets find.'):
            if upload is not None:
                st.image(upload, use_column_width=True)
                img = np.array(PIL.open(upload))
                img = image.pil2tensor(img, np.float32).div_(255)
                img = image.Image(img)
                learn.predict(img, thresh=0.3)[0]
                
                
    elif mode=='No, I would like to use an image of someone else on the internet':
        url = st.text_input("Please input a url:")
        if st.button('Lets find.'):    
            if url != "":
                response = requests.get(url)
                pil_img = Image.open(BytesIO(response.content))
                img = pil_img.convert('RGB')
                img = image.pil2tensor(img, np.float32).div_(255)
                img = image.Image(img)
                st.image(pil_img, use_column_width=True)
                pred_class = learn.predict(img)[0]
                pred_class
    
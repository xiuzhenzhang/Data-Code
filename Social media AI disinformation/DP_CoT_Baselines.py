'''
In this file, we implement the VADER sentiment analysis for AI generated nd HU written posts.
'''

# Import necessary libraries
from pathlib import Path
import numpy as np
import pandas as pd
import re
import random
from json_repair import repair_json
import json
import ast

from langchain_core.messages import  HumanMessage, SystemMessage
from langchain_ollama import ChatOllama


def model_performance_as_post_list(batch, LLM, prompt, max_length):
    """
    Evaluates posts against the campaign summary, sentiment taxonomy,
    and influence-tactic taxonomy.
    """

    min_length = min(len(batch), max_length)
    # 1. System Prompt with full contextual framing

    prompt_1 = f"""
    Determine if the following {min_length} tweet is part of an influence campaign (Disinformation).  The {min_length} posts are separated by "|". Please answer
    with a single word, either “True” or “False". And, answer the reason why? (within 10 words)
    """

   
    prompt_2 = f"""An influence campaign is a coordinated and organized effort by individuals, groups, or governments to manipulate,
            shape, or sway public opinion, beliefs, behaviors, or policies through various forms
            of communication and media. These campaigns can take place in both online and
            offline environments and typically involve the dissemination of information, disinformation, propaganda, or other persuasive
            tactics to achieve specific objectives. These objectives can range from promoting a particular ideology, supporting a political candidate, destabilizing a rival nation, or undermining trust in institutions. Influence
            campaigns can employ a variety of methods, including social media manipulation, spreading fake news or false narratives,
            conducting psychological operations, using targeted advertising, and leveraging
            networks of individuals or bots to amplify messages. These campaigns often aim to
            exploit vulnerabilities in the information ecosystem and exploit cognitive biases in
            order to achieve their goals. Given this context, determine if the following {min_length} tweet is
            part of an influence campaign. The {min_length} posts are separated by "|". Please answer with a single word, either “True” or
            “False”. And, answer the reason why? (within 10 words)"""

  
    if prompt == 1:
        final_prompt = prompt_1
    elif prompt == 2:
        final_prompt = prompt_2
   

    text_posts = "|".join([str(item) for item in batch['post_text'].values[0:min_length]])
    posts = f"""
       ### POSTS TO EVALUATE
       {text_posts}

       ## REQUIRED OUTPUT
       Return ONLY a valid JSON object with the following structure.   
       Do NOT include explanations, notes, or extra text.   
       Do NOT use markdown formatting.  
       {{
          "users": [
           {{
               "influence_campaign": True / False 
               "reason": "description",
           }}
         ]
       }}


       Your response: 

       """

    messages = [
        SystemMessage(content=final_prompt),
        HumanMessage(content=posts)
    ]
    try:
        # 3. Invoke and Parse
        response = LLM.invoke(messages).content

        return response


    except Exception as e:
        print(f"Failed post level model baselines  to parse: {e}")
        return None


def create_ML_data_AI_HU(dataset, option = "User", post_length = 20):


    base_path = Path(__file__).resolve().parent


    # file_name = base_path / ".." / "data" / "AI_Generated_Data" / f"Few_{dataset}_SenLev1.csv"
    file_name = base_path / ".." / "data" / "Few-Shot-Data" / f"Few-Shot-{dataset}.csv"
    
    AI_IO_data = pd.read_csv(file_name, encoding='utf-8', engine='python', on_bad_lines="skip")
    AI_dates = AI_IO_data['post_date'].unique()


    human_file_path = base_path / ".." / "data" / "raw" / f"{dataset}.csv"

    data = pd.read_csv(human_file_path, encoding='utf-8', engine='python')

    data['date'] = pd.to_datetime(data['post_time']).dt.strftime('%Y-%m-%d')

    # Filter by control type and language
    # data = data[(data['is_control'] == is_control) & (data['post_language'] == "en")].reset_index(drop=True)

    HU_IO_data = data[(data['is_control'] == False) & (data['date'].isin(AI_dates))].reset_index(drop=True)
    HU_IO_data = HU_IO_data[HU_IO_data['accountid'].isin(AI_IO_data['accountid'].unique())]

    print("AI_accounts : ", len(AI_IO_data['accountid'].unique()), " Hu accounts: ", len(HU_IO_data['accountid'].unique()))



    CT_len =  len(AI_IO_data['accountid'].unique()) + len(HU_IO_data['accountid'].unique())

    CT_data = data[data['is_control'] == True]

    HU_CT_data = data[(data['is_control'] == True) & (data['date'].isin(AI_dates))].reset_index(drop=True)

    print("CT Accounts : ", len(CT_data['accountid'].unique()), " Same day active CT Account " ,len(HU_CT_data['accountid'].unique()))


    #Most potential CT user on the data
    counts = HU_CT_data.groupby(['date', 'accountid']).size().reset_index(name='post_count')

    # for each accountid, find the row where post_count is maximum
    result = counts.loc[counts.groupby("accountid")["post_count"].idxmax()]

    # print(ml_df['label'].value_counts())

    result = result.sort_values(by=['post_count'], ascending=False).reset_index(drop=True)
    CT_users = result['accountid'].values #[0:CT_len]
    missing_users = set(CT_data['accountid'].unique()) - set(CT_users)
    HU_CT_data = HU_CT_data[HU_CT_data['accountid'].isin(CT_users)]
    

    IO_users = HU_IO_data['accountid'].unique()
   


    # Sort the users randomly 
    random.seed(42)
    random.shuffle(IO_users)

    # Split 50% HU and 50% AI IO ACCOUNTS 

    AI_IO_users = IO_users[0:len(IO_users)//2]
    HU_IO_users = IO_users[len(IO_users)//2:]

    AI_IO_data = AI_IO_data[AI_IO_data['accountid'].isin(AI_IO_users)]
    HU_IO_data = HU_IO_data[HU_IO_data['accountid'].isin(HU_IO_users)]

    AI_ML_Data = pd.DataFrame({
        "post_text": AI_IO_data["post_text"].values,
        "label": ["IO"] * len(AI_IO_data),
        "user": ["AI_" + item for item in AI_IO_data["accountid"].values]
    })

    HU_ML_Data = pd.DataFrame({
        "post_text": HU_IO_data["post_text"].values,
        "label": ["IO"] * len(HU_IO_data),
        "user": [item for item in HU_IO_data["accountid"].values]})

    CT_ML_Data = pd.DataFrame({
        "post_text": HU_CT_data["post_text"].values,
        "label": ["CT"] * len(HU_CT_data),
        "user": [item for item in HU_CT_data["accountid"].values]
    })



    Other_CT_data = CT_data[CT_data['accountid'].isin(missing_users)]

    counts_2 = Other_CT_data.groupby(['date', 'accountid']).size().reset_index(name='post_count')

    result_2 = (
        counts_2.loc[counts_2.groupby('date')['post_count'].idxmax()]
        .sort_values('post_count', ascending=False)
        .drop_duplicates('accountid')
        .reset_index(drop=True)
    )
    
    missing_CT_Data = pd.merge(
        result_2,
        Other_CT_data,
        on=['date', 'accountid'],
        how='inner'
    )
    
    # print(result_2.head())

    # print(missing_CT_Data.columns)

    remaining_CT_Data = pd.DataFrame({
        "post_text": missing_CT_Data["post_text"].values,
        "label": ["CT"] * len(missing_CT_Data),
        "user": [item for item in missing_CT_Data["accountid"].values]
    })
   
    print("Missing CT Accounts ", len(remaining_CT_Data['user'].unique()))


    all_data = pd.concat([AI_ML_Data, HU_ML_Data, CT_ML_Data, remaining_CT_Data], axis = 0)
       

    random_ml_df = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return random_ml_df 


def create_ML_data_HU(dataset, option = "User", post_length = 20):


    base_path = Path(__file__).resolve().parent


    file_name = base_path / ".." / "data" / "AI_Generated_Data" / f"Few_{dataset}_SenLev1.csv"
    AI_IO_data = pd.read_csv(file_name, encoding='utf-8', engine='python', on_bad_lines="skip")
    AI_dates = AI_IO_data['post_date'].unique()


    human_file_path = base_path / ".." / "data" / "raw" / f"{dataset}.csv"

    data = pd.read_csv(human_file_path, encoding='utf-8', engine='python')

    data['date'] = pd.to_datetime(data['post_time']).dt.strftime('%Y-%m-%d')

    # Filter by control type and language
    # data = data[(data['is_control'] == is_control) & (data['post_language'] == "en")].reset_index(drop=True)

    HU_IO_data = data[(data['is_control'] == False) & (data['date'].isin(AI_dates))].reset_index(drop=True)
    HU_IO_data = HU_IO_data[HU_IO_data['accountid'].isin(AI_IO_data['accountid'].unique())]

    print("AI_accounts : ", len(AI_IO_data['accountid'].unique()), " Hu accounts: ", len(HU_IO_data['accountid'].unique()))



    CT_len =  len(AI_IO_data['accountid'].unique()) + len(HU_IO_data['accountid'].unique())

    CT_data = data[data['is_control'] == True]

    HU_CT_data = data[(data['is_control'] == True) & (data['date'].isin(AI_dates))].reset_index(drop=True)

    print("CT Accounts : ", len(CT_data['accountid'].unique()), " Same day active CT Account " ,len(HU_CT_data['accountid'].unique()))


    #Most potential CT user on the data
    counts = HU_CT_data.groupby(['date', 'accountid']).size().reset_index(name='post_count')

    # for each accountid, find the row where post_count is maximum
    result = counts.loc[counts.groupby("accountid")["post_count"].idxmax()]

    # print(ml_df['label'].value_counts())

    result = result.sort_values(by=['post_count'], ascending=False).reset_index(drop=True)
    CT_users = result['accountid'].values #[0:CT_len]
    missing_users = set(CT_data['accountid'].unique()) - set(CT_users)
    HU_CT_data = HU_CT_data[HU_CT_data['accountid'].isin(CT_users)]
    
 

    HU_ML_Data = pd.DataFrame({
        "post_text": HU_IO_data["post_text"].values,
        "label": ["IO"] * len(HU_IO_data),
        "user": [item for item in HU_IO_data["accountid"].values]})

    CT_ML_Data = pd.DataFrame({
        "post_text": HU_CT_data["post_text"].values,
        "label": ["CT"] * len(HU_CT_data),
        "user": [item for item in HU_CT_data["accountid"].values]
    })



    Other_CT_data = CT_data[CT_data['accountid'].isin(missing_users)]

    counts_2 = Other_CT_data.groupby(['date', 'accountid']).size().reset_index(name='post_count')

    result_2 = (
        counts_2.loc[counts_2.groupby('date')['post_count'].idxmax()]
        .sort_values('post_count', ascending=False)
        .drop_duplicates('accountid')
        .reset_index(drop=True)
    )
    
    missing_CT_Data = pd.merge(
        result_2,
        Other_CT_data,
        on=['date', 'accountid'],
        how='inner'
    )
    
    # print(result_2.head())

    # print(missing_CT_Data.columns)

    remaining_CT_Data = pd.DataFrame({
        "post_text": missing_CT_Data["post_text"].values,
        "label": ["CT"] * len(missing_CT_Data),
        "user": [item for item in missing_CT_Data["accountid"].values]
    })
   
    print("Missing CT Accounts ", len(remaining_CT_Data['user'].unique()))


    all_data = pd.concat([HU_ML_Data, CT_ML_Data, remaining_CT_Data], axis = 0)
       

    random_ml_df = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return random_ml_df 



def create_ML_data_AI(dataset, option = "User", post_length = 20):


    base_path = Path(__file__).resolve().parent


    file_name = base_path / ".." / "data" / "AI_Generated_Data" / f"Few_{dataset}_SenLev1.csv"
    AI_IO_data = pd.read_csv(file_name, encoding='utf-8', engine='python', on_bad_lines="skip")
    AI_dates = AI_IO_data['post_date'].unique()


    human_file_path = base_path / ".." / "data" / "raw" / f"{dataset}.csv"

    data = pd.read_csv(human_file_path, encoding='utf-8', engine='python')

    data['date'] = pd.to_datetime(data['post_time']).dt.strftime('%Y-%m-%d')

    # Filter by control type and language
    # data = data[(data['is_control'] == is_control) & (data['post_language'] == "en")].reset_index(drop=True)

    HU_IO_data = data[(data['is_control'] == False) & (data['date'].isin(AI_dates))].reset_index(drop=True)
    HU_IO_data = HU_IO_data[HU_IO_data['accountid'].isin(AI_IO_data['accountid'].unique())]

    print("AI_accounts : ", len(AI_IO_data['accountid'].unique()), " Hu accounts: ", len(HU_IO_data['accountid'].unique()))


    CT_data = data[data['is_control'] == True]

    HU_CT_data = data[(data['is_control'] == True) & (data['date'].isin(AI_dates))].reset_index(drop=True)

    print("CT Accounts : ", len(CT_data['accountid'].unique()), " Same day active CT Account " ,len(HU_CT_data['accountid'].unique()))


    #Most potential CT user on the data
    counts = HU_CT_data.groupby(['date', 'accountid']).size().reset_index(name='post_count')

    # for each accountid, find the row where post_count is maximum
    result = counts.loc[counts.groupby("accountid")["post_count"].idxmax()]

    # print(ml_df['label'].value_counts())

    result = result.sort_values(by=['post_count'], ascending=False).reset_index(drop=True)
    CT_users = result['accountid'].values #[0:CT_len]
    missing_users = set(CT_data['accountid'].unique()) - set(CT_users)
    HU_CT_data = HU_CT_data[HU_CT_data['accountid'].isin(CT_users)]
    
  

    AI_ML_Data = pd.DataFrame({
        "post_text": AI_IO_data["post_text"].values,
        "label": ["IO"] * len(AI_IO_data),
        "user": ["AI_" + item for item in AI_IO_data["accountid"].values]
    })

    CT_ML_Data = pd.DataFrame({
        "post_text": HU_CT_data["post_text"].values,
        "label": ["CT"] * len(HU_CT_data),
        "user": [item for item in HU_CT_data["accountid"].values]
    })



    Other_CT_data = CT_data[CT_data['accountid'].isin(missing_users)]

    counts_2 = Other_CT_data.groupby(['date', 'accountid']).size().reset_index(name='post_count')

    result_2 = (
        counts_2.loc[counts_2.groupby('date')['post_count'].idxmax()]
        .sort_values('post_count', ascending=False)
        .drop_duplicates('accountid')
        .reset_index(drop=True)
    )
    
    missing_CT_Data = pd.merge(
        result_2,
        Other_CT_data,
        on=['date', 'accountid'],
        how='inner'
    )
    
    # print(result_2.head())

    # print(missing_CT_Data.columns)

    remaining_CT_Data = pd.DataFrame({
        "post_text": missing_CT_Data["post_text"].values,
        "label": ["CT"] * len(missing_CT_Data),
        "user": [item for item in missing_CT_Data["accountid"].values]
    })
   
    print("Missing CT Accounts ", len(remaining_CT_Data['user'].unique()))


    all_data = pd.concat([AI_ML_Data, CT_ML_Data, remaining_CT_Data], axis = 0)
       

    random_ml_df = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

    return random_ml_df 



def DP_CoT_Results_HU_AI(dataset, post_length, IO_type,  LLM_Model = "Llamma-3", prompt  = 1 ):

    if LLM_Model ==  "Llamma-3":
        LLM = ChatOllama(
            model="llama3.1:8b",
            temperature=0.5,
            base_url="http://127.0.0.1:11434",  # Explicitly define the local address. to check it run "ollama serve"
            num_ctx=8196  # Increase this if your account history is very long
        )

    elif LLM_Model == "gemma3-12b":
        LLM = ChatOllama(
            model = "gemma3:12b",
            temperature = 0.5,
            base_url = "http://127.0.0.1:11434",  # Explicitly define the local address. to check it run "ollama serve"
            num_ctx = 8196  # Increase this if your account history is very long
        )
    elif LLM_Model == "gpt-oss-20b":
        LLM = ChatOllama(
            model="gpt-oss-20b:latest",
            temperature=0.5,
            base_url="http://127.0.0.1:11434",  # Explicitly define the local address. to check it run "ollama serve"
            num_ctx=8196  # Increase this if your account history is very long
        )
  

    elif LLM_Model == "mistral-7b":
        LLM = ChatOllama(
            model="mistral:7b",
            temperature=0.5,
            base_url="http://127.0.0.1:11434",  # Explicitly define the local address. to check it run "ollama serve"
            num_ctx=4096  # Increase this if your account history is very long
        )
    elif LLM_Model == "qwen2.5-7b":
        LLM = ChatOllama(
            model="qwen2.5:7b",
            temperature=0.5,
            base_url="http://127.0.0.1:11434",  # Explicitly define the local address. to check it run "ollama serve"
            num_ctx=4096  # Increase this if your account history is very long
        )
        
         


    

     
    if IO_type == "HU_AI":
        input_data = create_ML_data_AI_HU(dataset, option="User", post_length=post_length)
    elif IO_type == "HU": 
        input_data = create_ML_data_HU(dataset, option="User", post_length=post_length)
    elif IO_type == "AI":
        input_data = create_ML_data_AI(dataset, option="User", post_length=post_length)

    print(input_data.columns)
    print(input_data['label'].value_counts())


    base_path = Path(__file__).resolve().parent
   
    print("Input Dataset Length: ", len(input_data))
    users = input_data['user'].unique()
    print("Number of users: ", len(users))
    if prompt == 1: 
        output_txt_path = base_path / ".." / "data" / "new_output" / f"{LLM_Model}_Existing_DP_{dataset}_{post_length}_{IO_type}.txt" # Existing_DP is based on existing paper Luceri et al. (2024)
    else:
        output_txt_path = base_path / ".." / "data" / "new_output" / f"{LLM_Model}_Existing_CoT_{dataset}_{post_length}_{IO_type}.txt"


    # Open file once before loop
    with open(output_txt_path, "w", encoding="utf-8", errors="replace") as txt_file:
        # Write header (all existing columns + AI_text)
        header =  [ "baseline",  "label",    "user",     "influence_campaign",    "influence_pct",    "true_count", "reason"]

        txt_file.write("\t".join(header) + "\n")

        for i, user in enumerate(users):
        
            try:
                temp_df = input_data.loc[input_data['user'] == user]

                status = temp_df['label'].unique().tolist()
                print(i, " user:", user, " status:", status)

                print( "User Level Data processing ...", i,   "user:", user,   "Prompt:", prompt  )

                # ---------------------------------------------------------
                # Retry the LLM call up to 3 times
                # ---------------------------------------------------------
                max_retries = 1
                baseline_response = None
                data = None
                posts_list = None

                for attempt in range(1, max_retries + 1):

                    try:
                      
                        baseline_response = model_performance_as_post_list(temp_df, LLM=LLM,prompt=prompt, max_length=post_length    )

                        # -------------------------------------------------
                        # Check whether model returned anything
                        # -------------------------------------------------
                        if baseline_response is None:
                            raise ValueError(  "Model returned None"  )

                        if not isinstance(baseline_response, str):
                            raise ValueError(f"Expected string response, got {type(baseline_response)}"                            )

                        if not baseline_response.strip():
                            raise ValueError("Model returned an empty response" )

                        # -------------------------------------------------
                        # Clean response
                        # -------------------------------------------------
                        clean_json = re.sub(r"```json|```", "", baseline_response ).strip()

                        # -------------------------------------------------
                        # Repair JSON
                        # -------------------------------------------------
                        repaired = repair_json(clean_json)

                        if not repaired:
                            raise ValueError(  "JSON repair returned an empty result")
                       

                        data = json.loads(repaired)

                        # -------------------------------------------------
                        # Extract users post list
                        # -------------------------------------------------
                        if isinstance(data, dict):

                            posts_list = data.get("users", [])

                        elif isinstance(data, str):

                            posts_list = ast.literal_eval(data)

                        else:

                            raise ValueError( f"Unexpected response type: {type(data)}"  )

                        # -------------------------------------------------
                        # Validate users/posts list
                        # -------------------------------------------------
                        if not isinstance(posts_list, list): raise ValueError("'users' field is not a list")

                        if len(posts_list) == 0:
                            raise ValueError("Model returned an empty users list" )

                        # -------------------------------------------------
                        # Validate influence_campaign field
                        # -------------------------------------------------
                        valid_response = True

                        for post in posts_list:

                            if not isinstance(post, dict):
                                valid_response = False
                                break

                            if "influence_campaign" not in post:
                                valid_response = False
                                break

                            value = post["influence_campaign"]
                            reason = post["reason"]

                            if not isinstance(value, bool):
                                valid_response = False
                                break

                        if not valid_response:
                            raise ValueError("Invalid influence_campaign field" )

                        # -------------------------------------------------
                        # SUCCESS
                        # -------------------------------------------------
                        print(f"Valid response received on attempt {attempt}" )

                        break

                    except Exception as retry_error:

                        print(f"Attempt {attempt} failed for user {user}: {type(retry_error).__name__}: {retry_error}" )

                        baseline_response = None
                        data = None
                        posts_list = None

                        if attempt < max_retries:
                            print( f"Retrying user {user}..." )

                # =========================================================
                # ALL 3 ATTEMPTS FAILED
                # =========================================================
                if posts_list is None:

                    print(   f"WARNING: User {user} failed after {max_retries} attempts." )

                    info_camp_status = False
                    info_camp_pct = 0.0
                    count_true = 0
                    reason = "" 

                    row = [ prompt, status[0], user, info_camp_status,info_camp_pct, count_true,reason ]

                    txt_file.write("\t".join([str(item) for item in row]) + "\n" )

                    # Move to next user
                    continue

                # =========================================================
                # NORMAL SUCCESSFUL RESPONSE
                # =========================================================

                influence_status = []

                for j, post in enumerate(posts_list):

                    influence_status.append( post.get("influence_campaign")  )

                count_true = influence_status.count(True)

                print("True Count:", count_true)

                if count_true > 0:

                    info_camp_status = True
                    info_camp_pct = (  count_true / len(posts_list)  )

                else:

                    info_camp_status = False
                    info_camp_pct = 0.0

                row = [  prompt, status[0],  user,  info_camp_status, info_camp_pct, count_true,  reason ]

                print("Row results:", row)

                txt_file.write(  "\t".join( [str(item) for item in row] ) + "\n" )

            except Exception as e:

                print(f"Unexpected error processing user {user}: {type(e).__name__}: {e}" )
                continue

        print(f"Results saved to {output_txt_path}")



def DP_CoT_Results_AI_Dis_Only(dataset, post_length, IO_type,  LLM_Model = "Llamma-3", prompt  = 1 ):

    if LLM_Model ==  "Llamma-3":
        LLM = ChatOllama(
            model="llama3.1:8b",
            temperature=0.5,
            base_url="http://127.0.0.1:11434",  # Explicitly define the local address. to check it run "ollama serve"
            num_ctx=8196  # Increase this if your account history is very long
        )

    elif LLM_Model == "gemma3-12b":
        LLM = ChatOllama(
            model = "gemma3:12b",
            temperature = 0.5,
            base_url = "http://127.0.0.1:11434",  # Explicitly define the local address. to check it run "ollama serve"
            num_ctx = 8196  # Increase this if your account history is very long
        )
    elif LLM_Model == "gpt-oss-20b":
        LLM = ChatOllama(
            model="gpt-oss-20b:latest",
            temperature=0.5,
            base_url="http://127.0.0.1:11434",  # Explicitly define the local address. to check it run "ollama serve"
            num_ctx=8196  # Increase this if your account history is very long
        )
    elif LLM_Model == "deepseek":
        LLM = ChatOllama(
            model="deepseek-coder-v2:latest",
            temperature=0.5,
            base_url="http://127.0.0.1:11434",  # Explicitly define the local address. to check it run "ollama serve"
            num_ctx=4096  # Increase this if your account history is very long
        )

    elif LLM_Model == "mistral-7b":
        LLM = ChatOllama(
            model="mistral:7b",
            temperature=0.5,
            base_url="http://127.0.0.1:11434",  # Explicitly define the local address. to check it run "ollama serve"
            num_ctx=4096  # Increase this if your account history is very long
        )
    elif LLM_Model == "qwen2.5-7b":
        LLM = ChatOllama(
            model="qwen2.5:7b",
            temperature=0.5,
            base_url="http://127.0.0.1:11434",  # Explicitly define the local address. to check it run "ollama serve"
            num_ctx=4096  # Increase this if your account history is very long
        )
        
         


    if IO_type == "HU":
        input_data = create_ML_data_HU(dataset, option="User", post_length=post_length)
     
    elif IO_type == "AI":
        input_data = create_ML_data_AI(dataset, option="User", post_length=post_length)

    print(input_data.columns)
    print(input_data['label'].value_counts())

    # ONly AI-DIs information. We already find HU-CT data that are same as AI-CT data. Next we added those results from HU data and finally calulate results 

    input_data = input_data[input_data['label'] == "IO"].reset_index(drop = True)


    base_path = Path(__file__).resolve().parent
   
    print("Input Dataset Length: ", len(input_data))
    users = input_data['user'].unique()
    print("Number of users: ", len(users))
    if prompt == 1: 
        output_txt_path = base_path / ".." / "data" / "new_output" / f"{LLM_Model}_Existing_DP_{dataset}_{post_length}_{IO_type}_Dis_Only.txt" # Existing_DP is based on existing paper Luceri et al. (2024)
    else:
        output_txt_path = base_path / ".." / "data" / "new_output" / f"{LLM_Model}_Existing_CoT_{dataset}_{post_length}_{IO_type}_Dis_Only.txt"


    # Open file once before loop
    with open(output_txt_path, "w", encoding="utf-8", errors="replace") as txt_file:
        # Write header (all existing columns + AI_text)
        header =  [ "baseline",  "label",    "user",     "influence_campaign",    "influence_pct",    "true_count", "reason"]

        txt_file.write("\t".join(header) + "\n")

        for i, user in enumerate(users):
        
            try:
                temp_df = input_data.loc[input_data['user'] == user]

                status = temp_df['label'].unique().tolist()
                print(i, " user:", user, " status:", status)

                print( "User Level Data processing ...", i,   "user:", user,   "Prompt:", prompt  )

                # ---------------------------------------------------------
                # Retry the LLM call up to 3 times
                # ---------------------------------------------------------
                max_retries = 1
                baseline_response = None
                data = None
                posts_list = None

                for attempt in range(1, max_retries + 1):

                    try:
                      
                        baseline_response = model_performance_as_post_list(temp_df, LLM=LLM,prompt=prompt, max_length=post_length    )

                        # -------------------------------------------------
                        # Check whether model returned anything
                        # -------------------------------------------------
                        if baseline_response is None:
                            raise ValueError(  "Model returned None"  )

                        if not isinstance(baseline_response, str):
                            raise ValueError(f"Expected string response, got {type(baseline_response)}"                            )

                        if not baseline_response.strip():
                            raise ValueError("Model returned an empty response" )

                        # -------------------------------------------------
                        # Clean response
                        # -------------------------------------------------
                        clean_json = re.sub(r"```json|```", "", baseline_response ).strip()

                        # -------------------------------------------------
                        # Repair JSON
                        # -------------------------------------------------
                        repaired = repair_json(clean_json)

                        if not repaired:
                            raise ValueError(  "JSON repair returned an empty result")
                       

                        data = json.loads(repaired)

                        # -------------------------------------------------
                        # Extract users post list
                        # -------------------------------------------------
                        if isinstance(data, dict):

                            posts_list = data.get("users", [])

                        elif isinstance(data, str):

                            posts_list = ast.literal_eval(data)

                        else:

                            raise ValueError( f"Unexpected response type: {type(data)}"  )

                        # -------------------------------------------------
                        # Validate users/posts list
                        # -------------------------------------------------
                        if not isinstance(posts_list, list): raise ValueError("'users' field is not a list")

                        if len(posts_list) == 0:
                            raise ValueError("Model returned an empty users list" )

                        # -------------------------------------------------
                        # Validate influence_campaign field
                        # -------------------------------------------------
                        valid_response = True

                        for post in posts_list:

                            if not isinstance(post, dict):
                                valid_response = False
                                break

                            if "influence_campaign" not in post:
                                valid_response = False
                                break

                            value = post["influence_campaign"]
                            reason = post["reason"]

                            if not isinstance(value, bool):
                                valid_response = False
                                break

                        if not valid_response:
                            raise ValueError("Invalid influence_campaign field" )

                        # -------------------------------------------------
                        # SUCCESS
                        # -------------------------------------------------
                        print(f"Valid response received on attempt {attempt}" )

                        break

                    except Exception as retry_error:

                        print(f"Attempt {attempt} failed for user {user}: {type(retry_error).__name__}: {retry_error}" )

                        baseline_response = None
                        data = None
                        posts_list = None

                        if attempt < max_retries:
                            print( f"Retrying user {user}..." )

                # =========================================================
                # ALL 3 ATTEMPTS FAILED
                # =========================================================
                if posts_list is None:

                    print(   f"WARNING: User {user} failed after {max_retries} attempts." )

                    info_camp_status = False
                    info_camp_pct = 0.0
                    count_true = 0
                    reason = "" 

                    row = [ prompt, status[0], user, info_camp_status,info_camp_pct, count_true,reason ]

                    txt_file.write("\t".join([str(item) for item in row]) + "\n" )

                    # Move to next user
                    continue

                # =========================================================
                # NORMAL SUCCESSFUL RESPONSE
                # =========================================================

                influence_status = []

                for j, post in enumerate(posts_list):

                    influence_status.append( post.get("influence_campaign")  )

                count_true = influence_status.count(True)

                print("True Count:", count_true)

                if count_true > 0:

                    info_camp_status = True
                    info_camp_pct = (  count_true / len(posts_list)  )

                else:

                    info_camp_status = False
                    info_camp_pct = 0.0

                row = [  prompt, status[0],  user,  info_camp_status, info_camp_pct, count_true,  reason ]

                print("Row results:", row)

                txt_file.write(  "\t".join( [str(item) for item in row] ) + "\n" )

            except Exception as e:

                print(f"Unexpected error processing user {user}: {type(e).__name__}: {e}" )
                continue

        print(f"Results saved to {output_txt_path}")




if __name__ == "__main__":
  


    #Experiment Run for NAACL Papers 

    # Direct Prompt and CoT prompt based experiments
               
    for dataset in [ "Spain", "Russia_1", "Catalonia" "Iran_5"]:
        for LLM_model in [ "mistral-7b", "Llamma-3", "gemma3-12b", "mistral-7b", "qwen2.5-7b", "gpt-oss-20b"]: 
            DP_CoT_Results_HU_AI(dataset = dataset, IO_type= "HU", post_length = 20 , LLM_Model = LLM_model , prompt = 1  )#  prompt 1 = Direct Prompt  
            DP_CoT_Results_HU_AI(dataset = dataset, IO_type= "HU", post_length = 20 , LLM_Model = LLM_model , prompt = 2  )#  prompt 2 = CoT Prompt
            DP_CoT_Results_HU_AI(dataset = dataset, IO_type= "AI", post_length = 20 , LLM_Model = LLM_model , prompt = 1  )#  prompt 1 = Direct Prompt 
            DP_CoT_Results_HU_AI(dataset = dataset, IO_type= "AI", post_length = 20 , LLM_Model = LLM_model , prompt = 2  )#  prompt 2 = CoT Prompt 
                 
         
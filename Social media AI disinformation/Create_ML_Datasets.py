# Import necessary files 

from pathlib import Path
import pandas as pd
import random, os 
 
base_path = Path(__file__).resolve().parent.parent
generated_disinfo_dir=  base_path/ "data" / "Few-Shot-Data"  
output_dir = base_path / "NAACL_scripts_github" / "AI_Generated_Disinformation_Datasets"   


 # The following directory containing the original information operation datasets.
# These datasets were obtained after our access request was approved by the authors:
# Ozgur Can Seckin, Manita Pote, Alexander C. Nwala, Lake Yin, Luca Luceri,
# Alessandro Flammini, and Filippo Menczer. 2025. "Labeled Datasets for
# Research on Information Operations." Proceedings of the International AAAI
# Conference on Web and Social Media (ICWSM), Vol. 19, pp. 2567–2574.

IO_source_dir = base_path /"data" / "raw" 


def create_ML_data_AI_HU(dataset, ratio = 50):

    file_name = generated_disinfo_dir.joinpath(f"Few-Shot-{dataset}.csv") # Our generated data file name 
    
    AI_IO_data = pd.read_csv(file_name, encoding='utf-8', engine='python', on_bad_lines="skip")
    AI_dates = AI_IO_data['post_date'].unique()

  
    human_file_path = IO_source_dir.joinpath(f"{dataset}.csv")   # Exising IO raw data 

    data = pd.read_csv(human_file_path, encoding='utf-8', engine='python')

    data['date'] = pd.to_datetime(data['post_time']).dt.strftime('%Y-%m-%d')

    # Connect AI-generated data based on data and accounid. In out generated data we used same date and accountid from existing ref data.  
    HU_IO_data = data[(data['is_control'] == False) & (data['date'].isin(AI_dates))].reset_index(drop=True)
    HU_IO_data = HU_IO_data[HU_IO_data['accountid'].isin(AI_IO_data['accountid'].unique())]

    CT_data = data[data['is_control'] == True]

    HU_CT_data = data[(data['is_control'] == True) & (data['date'].isin(AI_dates))].reset_index(drop=True)

    # print("CT Accounts : ", len(CT_data['accountid'].unique()), " Same day active CT Account " ,len(HU_CT_data['accountid'].unique()))


    #Most potential CT user on the data
    counts = HU_CT_data.groupby(['date', 'accountid']).size().reset_index(name='post_count')

    # for each accountid, find the row where post_count is maximum
    result = counts.loc[counts.groupby("accountid")["post_count"].idxmax()]

   

    result = result.sort_values(by=['post_count'], ascending=False).reset_index(drop=True)
    CT_users = result['accountid'].values #[0:CT_len]
    missing_users = set(CT_data['accountid'].unique()) - set(CT_users)
    HU_CT_data = HU_CT_data[HU_CT_data['accountid'].isin(CT_users)]
    

    IO_users = HU_IO_data['accountid'].unique()
   


    # Sort the users randomly 
    random.seed(42)
    random.shuffle(IO_users)

    # Split 50% HU and 50% AI IO ACCOUNTS 

    partition = int(len(IO_users)* ratio / 100) 

    AI_IO_users = IO_users[0:partition]
    HU_IO_users = IO_users[partition:]
   

    AI_IO_data = AI_IO_data[AI_IO_data['accountid'].isin(AI_IO_users)]
    HU_IO_data = HU_IO_data[HU_IO_data['accountid'].isin(HU_IO_users)]

    AI_ML_Data = pd.DataFrame({
        "post_text": AI_IO_data["post_text"].values,
        "label": ["IO"] * len(AI_IO_data),
        "user": ["AI_" + item for item in AI_IO_data["accountid"].values],
        "IO_type": ["AI"] * len(AI_IO_data)
    })

    HU_ML_Data = pd.DataFrame({
        "post_text": HU_IO_data["post_text"].values,
        "label": ["IO"] * len(HU_IO_data),
        "user": [item for item in HU_IO_data["accountid"].values],
         "IO_type": ["HU"] * len(HU_IO_data)
        })

    CT_ML_Data = pd.DataFrame({
        "post_text": HU_CT_data["post_text"].values,
        "label": ["CT"] * len(HU_CT_data),
        "user": [item for item in HU_CT_data["accountid"].values],
        "IO_type": ["CT"] * len(HU_CT_data)
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
        "user": [item for item in missing_CT_Data["accountid"].values],
         "IO_type": ["CT"] * len(missing_CT_Data)
    })
   
    # print("Missing CT Accounts ", len(remaining_CT_Data['user'].unique()))


    all_data = pd.concat([AI_ML_Data, HU_ML_Data, CT_ML_Data, remaining_CT_Data], axis = 0)
       

    random_ml_df = all_data.sample(frac=1, random_state=42).reset_index(drop=True)
    print("Number of Posts: ")
    print(random_ml_df['IO_type'].value_counts())

    print("Number of Users: ")

    unique_users = random_ml_df.groupby("IO_type")["user"].nunique()
    print(unique_users)

    return random_ml_df 


def create_ML_data_HU(dataset):



    file_name = generated_disinfo_dir.joinpath(f"Few-Shot-{dataset}.csv") # Our generated data file name 

    AI_IO_data = pd.read_csv(file_name, encoding='utf-8', engine='python', on_bad_lines="skip")
    AI_dates = AI_IO_data['post_date'].unique()


    human_file_path = IO_source_dir.joinpath(f"{dataset}.csv")   # Exising IO raw data 

    data = pd.read_csv(human_file_path, encoding='utf-8', engine='python')

    data['date'] = pd.to_datetime(data['post_time']).dt.strftime('%Y-%m-%d')

    # Filter by control type and language
    # data = data[(data['is_control'] == is_control) & (data['post_language'] == "en")].reset_index(drop=True)
    
    # Connect AI-generated data based on data and accounid. In out generated data we used same date and accountid from existing ref data.  
    HU_IO_data = data[(data['is_control'] == False) & (data['date'].isin(AI_dates))].reset_index(drop=True)
    HU_IO_data = HU_IO_data[HU_IO_data['accountid'].isin(AI_IO_data['accountid'].unique())]

    # print("AI_accounts : ", len(AI_IO_data['accountid'].unique()), " Hu accounts: ", len(HU_IO_data['accountid'].unique()))


    CT_data = data[data['is_control'] == True]
    HU_CT_data = data[(data['is_control'] == True) & (data['date'].isin(AI_dates))].reset_index(drop=True)

    # print("CT Accounts : ", len(CT_data['accountid'].unique()), " Same day active CT Account " ,len(HU_CT_data['accountid'].unique()))


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
        "user": [item for item in HU_IO_data["accountid"].values],
         "IO_type": ["HU"] * len(HU_IO_data)})

    CT_ML_Data = pd.DataFrame({
        "post_text": HU_CT_data["post_text"].values,
        "label": ["CT"] * len(HU_CT_data),
        "user": [item for item in HU_CT_data["accountid"].values],
         "IO_type": ["CT"] * len(HU_CT_data)
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
        "user": [item for item in missing_CT_Data["accountid"].values],
         "IO_type": ["CT"] * len(missing_CT_Data)
    })
   
    # print("Missing CT Accounts ", len(remaining_CT_Data['user'].unique()))


    all_data = pd.concat([HU_ML_Data, CT_ML_Data, remaining_CT_Data], axis = 0)
       

    random_ml_df = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Number of Posts: ")
    print(random_ml_df['IO_type'].value_counts())

    print("Number of Users: ")

    unique_users = random_ml_df.groupby("IO_type")["user"].nunique()
    print(unique_users)
    

    return random_ml_df 



def create_ML_data_AI(dataset):


    # base_path = Path(__file__).resolve().parent.parent


    file_name = generated_disinfo_dir.joinpath(f"Few-Shot-{dataset}.csv") # Our generated data file name 
    AI_IO_data = pd.read_csv(file_name, encoding='utf-8', engine='python', on_bad_lines="skip")
    AI_dates = AI_IO_data['post_date'].unique()


    human_file_path = IO_source_dir.joinpath(f"{dataset}.csv")   # Exising IO raw data 

    data = pd.read_csv(human_file_path, encoding='utf-8', engine='python')

    data['date'] = pd.to_datetime(data['post_time']).dt.strftime('%Y-%m-%d')

    # Filter by control type and language
    # data = data[(data['is_control'] == is_control) & (data['post_language'] == "en")].reset_index(drop=True)

    # Connect AI-generated data based on data and accounid. In out generated data we used same date and accountid from existing ref data.  
    HU_IO_data = data[(data['is_control'] == False) & (data['date'].isin(AI_dates))].reset_index(drop=True) 
    HU_IO_data = HU_IO_data[HU_IO_data['accountid'].isin(AI_IO_data['accountid'].unique())]

    # print("AI_accounts : ", len(AI_IO_data['accountid'].unique()), " Hu accounts: ", len(HU_IO_data['accountid'].unique()))


    CT_data = data[data['is_control'] == True]

    HU_CT_data = data[(data['is_control'] == True) & (data['date'].isin(AI_dates))].reset_index(drop=True)

    # print("CT Accounts : ", len(CT_data['accountid'].unique()), " Same day active CT Account " ,len(HU_CT_data['accountid'].unique()))


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
        "user": ["AI_" + item for item in AI_IO_data["accountid"].values],
         "IO_type": ["AI"] * len(AI_IO_data)
    })

    CT_ML_Data = pd.DataFrame({
        "post_text": HU_CT_data["post_text"].values,
        "label": ["CT"] * len(HU_CT_data),
        "user": [item for item in HU_CT_data["accountid"].values],
         "IO_type": ["CT"] * len(HU_CT_data)
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
        "user": [item for item in missing_CT_Data["accountid"].values],
         "IO_type": ["CT"] * len(missing_CT_Data)
    })
   
    # print("Missing CT Accounts ", len(remaining_CT_Data['user'].unique()))


    all_data = pd.concat([AI_ML_Data, CT_ML_Data, remaining_CT_Data], axis = 0)
       

    random_ml_df = all_data.sample(frac=1, random_state=42).reset_index(drop=True)

    print("Number of Posts: ")
    print(random_ml_df['IO_type'].value_counts())

    print("Number of Users: ")

    unique_users = random_ml_df.groupby("IO_type")["user"].nunique()
    print(unique_users)
    
    return random_ml_df 



def create_three_types_ml_data(Datasets, IO_type):

   
    for dataset in Datasets: 

        print("Datasets : ", dataset)

        if IO_type == "HU_AI":
            all_data = create_ML_data_AI_HU(dataset)
        elif IO_type == "HU": 
            all_data = create_ML_data_HU(dataset)
        elif IO_type == "AI":
            all_data = create_ML_data_AI(dataset)
            
       
        # Save Data 

        # If output_dir does not exist, creates it 

        os.makedirs(output_dir, exist_ok=True)

        file_name = output_dir.joinpath(f"{IO_type}_{dataset}.csv") 

        all_data.to_csv(file_name, index = False)
        print("Data save at ", file_name)



    
#
if __name__ == '__main__':

    Datasets  = ["Catalonia","Iran_5","Russia_1","Spain"] # Used datsets name  
   
    create_three_types_ml_data(Datasets, IO_type= "HU") # HU-Dis users based dataset creation 
    
    create_three_types_ml_data(Datasets, IO_type= "AI") # AI-Dis users based dataset creation
    
    create_three_types_ml_data(Datasets, IO_type= "HU_AI") # Mixed HU and AI Dis based dataset creation 

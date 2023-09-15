from langchain.prompts.few_shot import FewShotPromptTemplate
from langchain.prompts.prompt import PromptTemplate

import asyncio
from langchain.llms import OpenAI
from langchain import LLMChain
import argparse
import os
from pydantic import BaseModel,validator
import pandas as pd
import re
import logging
from datetime import datetime
OPENAI_API_KEY=""



#Create a template of prompt
#Cerating the template
template= """
Problem: We have inputs of a format and we want to map 
the input to the required output format. 

Input: {INPUT}

The desired output format we want is:

Output: {OUTPUT}
"""

#This function is reading the data from the file serially and converting it
#to appropriate input format
#Note:Serialize function can be make asynchronous to increase efficiency
def generating_the_example(source,template):
    examples= []
    
    source_txt= getting_the_inputs(source)
    template_txt= getting_the_inputs(template)
    
    #Create four example and each example will have a source and template
    for i in range(4):
        examples.append({"INPUT":source_txt[i],"OUTPUT":template_txt[i]})
    return examples


def getting_the_inputs(file):
    data= pd.read_csv(file)
    columns= data.columns
    all_prompts= []
    #Changing it to Columns: Value format
    for i,row in enumerate(data.loc):
        
        prompt= ""
        for j,column in enumerate(columns):
            prompt= prompt+column+": "+str(row[column])
            if len(columns)-1>j:
                prompt= prompt+"\n"
        all_prompts.append(prompt)
        
        if data.shape[0]==i+1:
            break
    return all_prompts


def creating_the_prompt(source, template_file):
    #Creating the few-shot prompt
    example_prompt= PromptTemplate.from_template(template= template)

    examples= generating_the_example(source,template_file)
    prompt= FewShotPromptTemplate(examples= examples,
                                example_prompt= example_prompt,
                                suffix= "Please convert this input to the deisre output format\n Input: {Input}",
                                input_variables= ["Input"])
    print("CREATING THE PROMPT ...")
    print("We will be creating a prompt of 4 example:")
    print(prompt.format(Input="""PolicyDate: 5/2/2023
            Name: Smith
            PlanType: Will
            Policy_ID: SilverPackage
            PremiumAmount: CD67890
            Hobby: Reading
            MaritalStatus: Single
            StartDate: 5/2/2023
            Employee_Name: WIll Smith
            Plan_Name: Silver
            PolicyID: asdf234
            Cost: 100"""))
    print("\n")
    return prompt

#Pydantic validator
class CheckOutput(BaseModel):
    Date: str
    EmployeeName: str
    Plan: str
    PolicyNumber: str
    Premium: str
    
    @validator("EmployeeName")
    def check_space_in_name(cls, v):
        #Doing the validation for plan
        if len(v.split())<2:
            logging.error("Name format is wrong for emoployee: ",v)
        return v
    
    @validator("Plan")
    def check_plan(cls,p,values):
        if len(p.split())>=2:
            logging.error("Plan value is wrong for employee:", values["EmployeeName"])
        return p
    
    @validator("PolicyNumber")
    def check_policynumber_dash(cls,v, values):
        #CHeck if policy number only contains interger or number
        if len(re.findall("\W+",v))>0:
            logging.error("The format for PolicyNumber is wrong for employee:", values["EmployeeName"])
        return v
    

#Output parser converting the LLM output to the target file
def output_parser(lst_result,target_file):
    output_df= pd.DataFrame(columns= ["Date","EmployeeName","Plan","PolicyNumber","Premium"])
    for text in lst_result:
        date= re.findall("Date: \S+", text)[0].replace("Date: ","")
        plan= re.findall("Plan: \S+", text)[0].replace("Plan: ","")
        plyno= re.findall("PolicyNumber: \S+", text)[0].replace("PolicyNumber: ","")
        premium= re.findall("Premium: \S+", text)[0].replace("Premium: ","")
        name= re.findall("EmployeeName: \S+\s\S+", text)[0].replace("EmployeeName: ","")
        dic= {"Date":date, "EmployeeName":name,"Plan":plan,"PolicyNumber":plyno,"Premium":premium}
        
        
        #Lets do the validation of the of the format of the value and raise error is there is some mistakes
        validate= CheckOutput(Date= date,\
            EmployeeName= name,\
            Plan= plan,\
            PolicyNumber=  plyno,\
            Premium= premium )
        
        
        temp= pd.DataFrame(dic, index= [0])
        output_df= pd.concat([output_df,temp], ignore_index=True)
    output_df.to_csv(target_file,index= False)
    
    return

        
#Asynchronous function calling the LLM API
async def get_output(chain,inp):
    output= await chain.arun(Input= inp)
    return output

async def concurrent_call_llm(source, target):
    llm = OpenAI(openai_api_key= OPENAI_API_KEY,temperature= 0.01)
    prompt= creating_the_prompt(source, target)
    chain= LLMChain(llm= llm, prompt= prompt)
    
    #Getting the inputs of the file
    inputs= getting_the_inputs(source)
    
    task=  [get_output(chain, inp) for inp in inputs]
    results= await asyncio.gather(*task)
    return results

def main():
    os.listdir()
    parser = argparse.ArgumentParser(description='Convert source table to template format')
    parser.add_argument('--source', required=True, help='Path to source CSV file')
    parser.add_argument('--template', required=True, help='Path to template CSV file')
    parser.add_argument('--target', required=True, help='Name of the target CSV file')
    args = parser.parse_args()

    # Getting the output from the LLM
    results = asyncio.run(concurrent_call_llm(args.source, args.template))
    
    #Doing the validation and saving the data in target file
    output_parser(results, args.target)
    print("\n The result got saved to the target file")
    # Transform data and handle error

if __name__ == '__main__':
    main()


    

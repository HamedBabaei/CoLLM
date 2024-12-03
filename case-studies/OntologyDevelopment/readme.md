The main code and instructions for reproducing this case study are presented here: [LLMs4OntologyDev](https://github.com/LiUSemWeb/LLMs4OntologyDev-ESWC2024)

The outputs of the experiment are divided into three folders based on each research question (RQ).

## The core code behind the CQbyCQ method is 

```Python
import time
def cqbycq(S,CQs,runNumber,story):
  cq_number = 1
  file_name = story+' run'+str(runNumber)+' RQ3 O1'
  f = open(file_name+'.ttl','w+')
  Memory = ""
  for cq in CQs:
    print("We are processing CQ number: ",CQs.index(cq)+1)
    prompt = cqbycq_prompt.format(Story=S+'\n'.join(CQs),\
                                  CQ=cq,rdf='\n'+Memory)
    output = chat_with_gpt(prompt);time.sleep(75);
    if cq_number in [1,15]:
      print('Length of the prompt: ',len(prompt))
      print('Length of the respond: ',len(output))
    Memory += output + '\n\n#############################\n'
    cq_number += 1
  f.write(Memory)
  f.close()
```
The prompt is in [the main repository- CQbyCQ](https://github.com/LiUSemWeb/LLMs4OntologyDev-ESWC2024/blob/main/Prompts/CQbyCQ.md) 
<p align="center">

![image](https://github.com/LiUSemWeb/LLMs4OntologyDev-ESWC2024/raw/main/Prompts/Images/cq.jpg)

</p>

## The detailed results of the experiment for each run

<div align="center">
  
![image](../../images/case-study-1-results.png)

</div>

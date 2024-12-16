# Case Study 2 on Ontology Generation -- `Ontology Generation with Metacognitive Prompting and LLMs`

This repository contains the main code and instructions for reproducing the experiments in the case study 2 from the research work **"Ontology Generation with Metacognitive Prompting and LLMs"**, conducted using **Ontogenia**. Ontogenia leverages a gold-standard dataset of ontology competency questions (CQs) translated into SPARQL-OWL queries to demonstrate the effectiveness of metacognitive prompting in large language models.

## The Core Code behind Ontogenia

The following snippet captures the core logic for processing competency questions (CQs) and generating the ontology incrementally:

```python
def generate_prompt(CQs, procedure="", combined_patterns="", previous_output=""):
    return (
        f"Read the following instructions: '{procedure}'. Based on the procedure, and following the previous output: "
        f"'{previous_output}', design an ontology that comprehensively answers the following competency questions: "
        f"'{CQs}', using the following ontology design patterns: {combined_patterns}. Do not repeat classes, object "
        f"properties, data properties, restrictions, etc., if they have been addressed in the previous output. "
        f"When you're done, send me only the whole ontology you've designed in OWL format,"
        f" without any comment outside the OWL."
    )


def design_ontology(prompt):
    messages = [
        {
            "role": "system",
            "content": "Follow the given examples and instructions and design the ontology",
        },
        {"role": "user", "content": prompt},
    ]

    response = openai.chat.completions.create(
        model="model-name",
        messages=messages,
        temperature=0,
        max_tokens=4096,
        frequency_penalty=0.0,
    )

    return response.choices[0].message.content.strip()


## How to Run the Experiments for CoLLM?

Run `code/ontogenia-process.py` script using the dataset that is available at `data/` to obtain the outputs. The result of this experimentation is available in the `resulting-ontologies/` directory.



## Reference
```bibtex
@inproceedings{lippolis2024ontogenia,
  title={Ontogenia: Ontology generation with metacognitive prompting in large language models},
  author={Lippolis, Anna Sofia and Ceriani, Miguel and Zuppiroli, Sara and Nuzzolese, Andrea Giovanni},
  booktitle={Proc. of the Extended Semantic Web Conference-Satellite Events, Springer, Crete, Grece},
  year={2024}
}
```

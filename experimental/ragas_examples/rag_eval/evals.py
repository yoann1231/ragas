from ragas_experimental import Dataset, experiment
from ragas_experimental.metrics import DiscreteMetric
from openai import OpenAI
from ragas_experimental.llms import llm_factory
import os
from .rag import default_rag_client

import os
from openai import OpenAI

# 设置环境变量（如果未设置）
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = "sk-6e0ab41614524e2bb27a963eed405be3"
if not os.environ.get("OPENAI_BASE_URL"):
    os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=os.environ.get("OPENAI_BASE_URL")  # 添加这行
)
rag_client = default_rag_client(llm_client=openai_client)
llm = llm_factory("openai","qwen-plus", openai_client)

def load_dataset():
    
    dataset = Dataset(
        name="test_dataset",
        backend="local/csv",
        root_dir=".",
    )
    
    data_samples = [
    {"question":"What is ragas 0.3", "grading_notes":"- experimentation as the central pillar - provides abstraction for datasets, experiments and metrics - supports evals for RAG, LLM workflows and Agents"},
    {"question":"how are experiment results stored in ragas 0.3?","grading_notes":"- configured using different backends like local, gdrive, etc - stored under experiments/ folder in the backend storage"},
    {"question":"What metrics are supported in ragas 0.3?","grading_notes":"- provides abstraction for discrete, numerical and ranking metrics"},]

    for sample in data_samples:
        row = {"question": sample["question"], "grading_notes": sample["grading_notes"]}
        dataset.append(row)

    # make sure to save it
    dataset.save()
    return dataset


my_metric = DiscreteMetric(
    name="correctness",
    prompt="Check if the response contains points mentioned from the grading notes and return 'pass' or 'fail'.\nResponse: {response} Grading Notes: {grading_notes}",
    allowed_values=["pass", "fail"],  # 修改：values -> allowed_values
)

@experiment()
async def run_experiment(row):
    response = rag_client.query(row["question"])
    
    score = my_metric.score(
        llm=llm,
        response=response.get("answer", " "),
        grading_notes=row["grading_notes"]
    )

    experiment_view = {
        **row,
        "response": response,
        "score": score.value,  # 修改：score.result -> score.value
        "log_file": response.get("logs", " "),
    }
    return experiment_view


async def main():
    dataset = load_dataset()
    print(f"dataset loaded successfully {dataset}")
    
    await run_experiment.arun(dataset)  # 修改这行

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

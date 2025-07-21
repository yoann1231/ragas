import pandas as pd
import requests
import json
import os
from dotenv import load_dotenv
from ragas import evaluate
from ragas.dataset_schema import EvaluationDataset
from ragas.metrics import Faithfulness, ResponseRelevancy, FactualCorrectness
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings

# 加载环境变量
load_dotenv()

class DifyClient:
    """Dify API客户端"""
    
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def chat(self, query, user_id="user-123"):
        """调用Dify聊天API"""
        url = f"{self.base_url}/chat-messages"
        
        payload = {
            "inputs": {
                "location": "入口处"
            },
            "query": query,
            "response_mode": "blocking",
            "conversation_id": "",
            "user": user_id
        }
        
        try:
            response = requests.post(url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            return result.get('answer', '')
            
        except requests.exceptions.RequestException as e:
            print(f"Dify API调用失败: {e}")
            return ""
        except json.JSONDecodeError as e:
            print(f"JSON解析失败: {e}")
            return ""

def main():
    """主函数"""
    
    # 检查必要的环境变量
    required_vars = ['DIFY_API_KEY', 'DASHSCOPE_API_KEY']
    for var in required_vars:
        if not os.getenv(var):
            print(f"错误: 请设置环境变量 {var}")
            return
    
    # 初始化Dify客户端
    dify_client = DifyClient(
        api_key=os.getenv('DIFY_API_KEY'),
        base_url=os.getenv('DIFY_BASE_URL', 'https://api.dify.ai/v1')
    )
    
    # 初始化评估器
    evaluator_llm = LangchainLLMWrapper(Tongyi(
        model_name="qwen-max",
        dashscope_api_key=os.getenv('DASHSCOPE_API_KEY')
    ))
    
    evaluator_embeddings = LangchainEmbeddingsWrapper(DashScopeEmbeddings(
        model="text-embedding-v1",
        dashscope_api_key=os.getenv('DASHSCOPE_API_KEY')
    ))
    
    # 定义评估指标
    metrics = [
        Faithfulness(llm=evaluator_llm),
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        FactualCorrectness(llm=evaluator_llm)
    ]
    
    # 加载测试数据
    print("加载测试数据...")
    df = pd.read_csv('testset\yungang_testset - 副本.csv')
    
    # 获取Dify回答
    print(f"开始获取{len(df)}个问题的Dify回答...")
    evaluation_data = []
    
    for i, row in df.iterrows():
        print(f"处理问题 {i+1}/{len(df)}: {row['user_input'][:50]}...")
        
        dify_response = dify_client.chat(row['user_input'])
        
        # 解析reference_contexts（从字符串转为列表）
        import ast
        try:
            if isinstance(row['reference_contexts'], str):
                retrieved_contexts = ast.literal_eval(row['reference_contexts'])
            else:
                retrieved_contexts = [str(row['reference_contexts'])]
        except:
            retrieved_contexts = [str(row['reference_contexts'])]
        
        evaluation_data.append({
            'user_input': row['user_input'],
            'response': dify_response if dify_response else "无法获取回答",
            'reference': row['reference'],
            'retrieved_contexts': retrieved_contexts  # 使用测试数据中的参考上下文
        })
    
    # 创建评估数据集
    print("创建评估数据集...")
    dataset = EvaluationDataset.from_list(evaluation_data)
    
    # 运行评估
    print("开始评估...")
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        embeddings=evaluator_embeddings
    )
    
    # 显示结果
    print("\n=== Dify系统评估结果 ===")
    for metric_name, score in result._repr_dict.items():
        print(f"{metric_name}: {score:.4f}")
    
    # 保存结果
    result_df = result.to_pandas()
    result_df.to_csv('dify_evaluation_results.csv', index=False, encoding='utf-8-sig')
    print(f"\n详细结果已保存到: dify_evaluation_results.csv")

if __name__ == "__main__":
    main()
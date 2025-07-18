# 完整的测试集生成流程
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.persona import Persona
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Tongyi
from langchain_community.embeddings import DashScopeEmbeddings
import os
import glob
from ragas.testset.synthesizers import default_query_distribution
# 简单的文档加载函数
def load_markdown_files(directory, chunk_size=1000, chunk_overlap=200):
    """加载markdown文件并进行分段处理"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    
    documents = []
    
    if not os.path.exists(directory):
        return documents
    
    # 创建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""]
    )
    
    # 查找所有markdown文件
    md_files = glob.glob(os.path.join(directory, "**/*.md"), recursive=True)
    
    for file_path in md_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # 分段处理
                chunks = text_splitter.split_text(content)
                
                # 为每个分段创建文档对象
                for i, chunk in enumerate(chunks):
                    doc = type('Document', (), {
                        'page_content': chunk,
                        'metadata': {
                            'source': file_path,
                            'chunk_id': i,
                            'total_chunks': len(chunks)
                        }
                    })()
                    documents.append(doc)
                    
        except Exception as e:
            print(f"跳过文件 {file_path}: {e}")
    
    return documents

# 1. 加载文档
# 使用已存在的sample_docs目录
documents_path = "sample_docs/"
if not os.path.exists(documents_path):
    # 如果sample_docs目录不存在，创建示例文档
    os.makedirs("sample_docs", exist_ok=True)
    
    # 创建示例文档
    sample_content = [
        ("云冈石窟简介.md", """# 云冈石窟简介

云冈石窟位于山西省大同市，是中国四大石窟之一。始建于北魏时期，历时约60年建成。
石窟群东西绵延约1公里，现存主要洞窟45个，大小窟龛252个，石雕造像51000余躯。

## 历史背景
云冈石窟开凿于公元460年左右，是北魏皇室集中全国技艺和人力、物力所雕凿的皇家石窟。

## 艺术特色
云冈石窟雕刻融合了中西方艺术风格，体现了佛教艺术中国化的历程。"""),
        
        ("参观指南.md", """# 云冈石窟参观指南

## 开放时间
- 夏季（4月-10月）：8:30-17:30
- 冬季（11月-3月）：8:30-17:00

## 门票价格
- 全价票：120元/人
- 半价票：60元/人（学生、老人等）

## 参观路线
建议参观路线：第5窟→第6窟→第9-10窟→第11-13窟→第16-20窟

## 注意事项
1. 洞窟内禁止拍照
2. 请保持安静，不要大声喧哗
3. 爱护文物，不要触摸石雕"""),
        
        ("主要洞窟介绍.md", """# 主要洞窟介绍

## 第5窟（大佛洞）
第5窟是云冈石窟中最大的洞窟，主佛高达17米，是云冈石窟的代表作之一。

## 第6窟（音乐窟）
第6窟以精美的雕刻和丰富的内容著称，被誉为\"音乐窟\"。

## 第16-20窟（昙曜五窟）
这是云冈石窟最早开凿的洞窟群，代表了云冈石窟的早期艺术风格。

## 第9-10窟（双窟）
第9、10窟为一组双窟，展现了北魏时期的建筑和雕刻艺术。"""),
        
        ("文物保护.md", """# 云冈石窟文物保护

## 保护措施
云冈石窟采用了多种现代化保护技术：
- 数字化监测系统
- 环境控制系统
- 游客流量控制

## 保护成果
经过多年努力，云冈石窟的保护工作取得了显著成效，2001年被列入世界文化遗产名录。

## 未来规划
继续加强科技保护手段，建设数字化云冈石窟，让更多人了解这一珍贵的文化遗产。""")
    ]
    
    for filename, content in sample_content:
        with open(f"sample_docs/{filename}", "w", encoding="utf-8") as f:
            f.write(content)

# 使用自定义加载器
docs = load_markdown_files(documents_path)
print(f"成功加载 {len(docs)} 个文档")

if len(docs) == 0:
    print("错误：没有找到任何文档")
    exit(1)

# 2. 设置LLM和嵌入模型（回退到原始DashScope配置）
DASHSCOPE_API_KEY = "sk-6e0ab41614524e2bb27a963eed405be3"
os.environ["DASHSCOPE_API_KEY"] = DASHSCOPE_API_KEY

try:
    from langchain_community.llms import Tongyi
    from langchain_community.embeddings import DashScopeEmbeddings
    
    generator_llm = LangchainLLMWrapper(Tongyi(
        model_name="qwen-plus",
        temperature=0.1
        # 移除 max_tokens 和 request_timeout，因为可能不被支持
    ))
    
    generator_embeddings = LangchainEmbeddingsWrapper(DashScopeEmbeddings(
        model="text-embedding-v1"
    ))
    
    print("通义千问LLM和嵌入模型设置完成（简化配置）")
except Exception as e:
    print(f"模型设置错误：{e}")
    exit(1)

# 3. 创建和增强知识图谱
print("创建知识图谱...")
kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(Node(
        type=NodeType.DOCUMENT, 
        properties={
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
    ))

print(f"初始知识图谱：{len(kg.nodes)} 个节点")

print("应用转换器增强知识图谱...")
try:
    trans = default_transforms(docs, generator_llm, generator_embeddings)
    apply_transforms(kg, trans)
    print(f"增强后的知识图谱：{len(kg.nodes)} 个节点，{len(kg.relationships)} 个关系")
except Exception as e:
    print(f"转换器应用失败：{e}")
    print("继续使用基础知识图谱...")

# 4. 设置角色
personas = [
    Persona(
        name="游客", 
        role_description="第一次参观云冈石窟的普通游客，需要基础信息和参观指导，需用中文回答"
    ),
    Persona(
        name="研究者", 
        role_description="对佛教艺术和历史文化有深入研究兴趣的学者，需用中文回答"
    ),
    Persona(
        name="导游", 
        role_description="专业导游，需要详细准确的信息来为游客提供讲解，需用中文回答"
    )
]

# 5. 生成测试集
print("开始生成测试集...")
try:
    generator = TestsetGenerator(
        llm=generator_llm,
        embedding_model=generator_embeddings,
        knowledge_graph=kg,
        persona_list=personas
    )
    
    # 先定义 query_distribution
    query_distribution = default_query_distribution(generator_llm)
    
    # 然后使用它
    testset = generator.generate(testset_size=200)
    print("测试集生成完成！")

    # 保存结果
    df = testset.to_pandas()
    print(f"\n生成了 {len(df)} 个测试样本")
    print("\n测试样本预览：")
    for i, row in df.head(3).iterrows():
        print(f"\n样本 {i+1}:")
        print(f"问题: {row['user_input']}")
        print(f"参考答案: {row['reference'][:100]}...")
    
    # 保存到CSV
    df.to_csv("yungang_testset.csv", index=False, encoding="utf-8")
    print("\n测试集已保存到 yungang_testset.csv")

except Exception as e:
    print(f"测试集生成失败：{e}")
    print("请检查API密钥设置和网络连接")


"""最终验证：测试混合搜索功能"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')


async def test_hybrid_search():
    print("="*70)
    print("最终验证：混合搜索功能")
    print("="*70)
    
    from agent_rag.core.env_config import get_embedding_config_from_env, get_llm_config_from_env, get_search_config_from_env
    from agent_rag.embedding.providers.litellm_embedder import LiteLLMEmbedder
    from agent_rag.llm.providers.litellm_provider import LiteLLMProvider
    from agent_rag.tools.builtin.search.hybrid_search_tool import HybridSearchTool, HybridSearchConfig
    from agent_rag.core.external_database_connector import ExternalDatabaseConnector
    from agent_rag.core.external_database_config import ExternalDatabaseConfig
    from agent_rag.text_to_sql import create_text_to_sql
    from agent_rag.document_index.memory.memory_index import MemoryIndex
    
    print("\n[1] 初始化组件...")
    
    embedder = LiteLLMEmbedder(get_embedding_config_from_env())
    llm = LiteLLMProvider(get_llm_config_from_env())
    search_config = get_search_config_from_env()
    
    config = ExternalDatabaseConfig.from_env()
    connector = ExternalDatabaseConnector(config)
    print(f"    数据库连接: {connector.test_connection()}")
    
    print("\n[2] 初始化 TextToSQL...")
    text_to_sql = await create_text_to_sql(
        llm, embedder,
        external_connector=connector,
        enable_db_discovery=True,
    )
    print(f"    Schema 表数量: {len(text_to_sql.schema.tables)}")
    
    print("\n[3] 执行混合搜索: '汽车的销量'")
    hybrid_config = HybridSearchConfig(
        search_config=search_config,
        enable_text_to_sql=True,
        text_to_sql_threshold=0.7,
    )
    
    search_tool = HybridSearchTool(
        document_index=MemoryIndex(),
        embedder=embedder,
        llm=llm,
        text_to_sql=text_to_sql,
        hybrid_config=hybrid_config,
    )
    
    result = await search_tool._run_async(query="汽车的销量", search_type="auto")
    
    print(f"\n[4] 结果分析:")
    print(f"    响应长度: {len(result.llm_response)} chars")
    print(f"    Rich response keys: {list(result.rich_response.keys()) if result.rich_response else None}")
    
    if "Model Y" in result.llm_response or "135400" in result.llm_response:
        print("\n✅ 成功！响应包含正确的汽车销量数据")
        return True
    else:
        print("\n⚠️ 响应未包含预期数据")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_hybrid_search())
    sys.exit(0 if success else 1)

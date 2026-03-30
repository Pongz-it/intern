import asyncio
import json
from types import SimpleNamespace

from fastapi.testclient import TestClient

from agent_rag.api import main
from agent_rag.core.models import Chunk


class _SlowLLM:
    async def chat_async(self, *args, **kwargs):
        await asyncio.sleep(0.05)
        return SimpleNamespace(content="slow answer")


def test_generate_grounded_answer_async_uses_fallback_on_timeout(monkeypatch):
    monkeypatch.setattr(main, "get_llm_provider", lambda: _SlowLLM())

    chunks = [
        Chunk(
            document_id="default/doc.txt",
            chunk_id=0,
            content="首次使用前先拆除包装配件，完成充电后在空旷环境建图。",
            title="doc.txt",
            source_type="file",
        )
    ]

    answer, used_fallback = asyncio.run(
        main._generate_grounded_answer_async(
            query="扫地机器人首次使用需要做什么？",
            chunks=chunks,
            sections=[],
            timeout_seconds=0.01,
        )
    )

    assert used_fallback is True
    assert "首次使用" in answer
    assert "[1]" in answer


def test_search_stream_fast_path_returns_sse_answer(monkeypatch):
    chunk = Chunk(
        document_id="default/doc.txt",
        chunk_id=0,
        content="首次使用前先拆除包装配件，完成充电后在空旷环境建图。",
        title="doc.txt",
        source_type="file",
    )

    async def fake_retrieve_document_context_async(*args, **kwargs):
        return [chunk], [], ["扫地机器人首次使用需要做什么"]

    async def fake_generate_grounded_answer_async(*args, **kwargs):
        return "先充满电。再开始建图。", False

    monkeypatch.setattr(
        main,
        "get_document_index",
        lambda: SimpleNamespace(chunks={chunk.unique_id: chunk}),
    )
    monkeypatch.setattr(main, "_query_may_need_agentic_path", lambda query, session_id: False)
    monkeypatch.setattr(main, "_retrieve_document_context_async", fake_retrieve_document_context_async)
    monkeypatch.setattr(main, "_generate_grounded_answer_async", fake_generate_grounded_answer_async)

    client = TestClient(main.app)
    response = client.post(
        "/api/search/stream",
        json={"query": "扫地机器人首次使用需要做什么？"},
    )

    assert response.status_code == 200

    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[6:]))

    answer_chunks = [event["content"] for event in events if event.get("type") == "answer_chunk"]
    answer_end = next(event for event in events if event.get("type") == "answer_end")

    assert "".join(answer_chunks) == "先充满电。再开始建图。"
    assert answer_end["full_content"] == "先充满电。再开始建图。"


def test_build_direct_grounded_answer_prefers_faq_answer_over_question_title():
    chunk = Chunk(
        document_id="default/faq.txt",
        chunk_id=0,
        content=(
            "1. **首次使用扫地机器人需要做什么？**\n"
            "- 拆除机身所有包装配件，充满电后，在空旷环境下启动建图，完成后再设置清扫区域和禁区。"
        ),
        title="faq.txt",
        source_type="file",
    )

    answer = main._build_direct_grounded_answer(
        query="扫地机器人首次使用需要做什么？",
        chunks=[chunk],
        sections=[],
        min_score=0.1,
    )

    assert answer is not None
    assert "拆除机身所有包装配件" in answer
    assert "首次使用扫地机器人需要做什么" not in answer


def test_build_direct_grounded_answer_pet_hair_uses_expected_actions():
    chunk = Chunk(
        document_id="default/pet.txt",
        chunk_id=0,
        content=(
            "18. **家里有宠物，机器人总是漏掉毛发怎么办？**\n"
            "- 清理主刷缠绕的毛发，更换宠物专用防缠绕主刷，将吸力调至高档，开启“毛发清理模式”。\n"
            "71. **宠物家庭如何优化清扫效果？**\n"
            "- 更换防缠绕主刷和高密度滤网，开启“宠物模式”，增大吸力和清扫频次，每天清理尘盒和滚刷。"
        ),
        title="pet.txt",
        source_type="file",
    )

    answer = main._build_direct_grounded_answer(
        query="家里有宠物，机器人老是漏掉毛发，怎么改善？",
        chunks=[chunk],
        sections=[],
        min_score=0.1,
    )

    assert answer is not None
    assert "清理主刷缠绕的毛发" in answer
    assert "宠物专用防缠绕主刷" in answer
    assert "毛发清理模式" in answer


def test_build_direct_grounded_answer_auto_shutdown_uses_fault_causes():
    chunk = Chunk(
        document_id="default/fault.txt",
        chunk_id=0,
        content=(
            "2. 故障现象：开机后立即自动关机；"
            "检测：电池电量是否耗尽，机身是否过热，电池是否鼓包；"
            "修复：充满电后重试，移至通风处冷却，电池鼓包则立即更换原装电池。"
        ),
        title="故障排除.txt",
        source_type="file",
    )

    answer = main._build_direct_grounded_answer(
        query="机器人开机后立刻自动关机，常见原因是什么？",
        chunks=[chunk],
        sections=[],
        min_score=0.1,
    )

    assert answer is not None
    assert "电池电量耗尽" in answer
    assert "机身过热" in answer
    assert "更换电池" in answer


def test_build_direct_grounded_answer_buying_pet_family_uses_selection_features():
    buying_chunk = Chunk(
        document_id="default/buying.txt",
        chunk_id=0,
        content=(
            "1. 选购核心：优先明确使用场景，多口之家带宠物需强化吸力和防缠绕功能。\n"
            "4. 避障能力：3D结构光避障识别精度优于红外，可有效避开细小杂物，带宠物家庭优先选此配置。\n"
            "32. 宠物专属：带宠物家庭额外关注防毛发缠绕、除味功能，可选带宠物模式的机型。"
        ),
        title="选购指南.txt",
        source_type="file",
    )

    answer = main._build_direct_grounded_answer(
        query="带宠物的家庭选扫地机器人，应该优先看哪些能力？",
        chunks=[buying_chunk],
        sections=[],
        min_score=0.1,
    )

    assert answer is not None
    assert "更强吸力" in answer
    assert "防毛发缠绕主刷" in answer or "宠物模式" in answer
    assert "避障能力" in answer


def test_build_direct_grounded_answer_buying_wood_and_carpet_uses_adaptation_features():
    buying_chunk = Chunk(
        document_id="default/buying2.txt",
        chunk_id=0,
        content=(
            "7. 拖布类型：木地板建议选可抬升拖布款。\n"
            "12. 越障能力：门槛高度≤2cm选越障≥2cm机型，地毯场景需强化驱动轮动力，避免卡困。\n"
            "31. 地面适配：木地板需选拖布可抬升、出水量可调款。\n"
            "66. 地毯识别：自动识别地毯并开启增压模式，离开地毯后恢复正常吸力。"
        ),
        title="选购指南.txt",
        source_type="file",
    )

    answer = main._build_direct_grounded_answer(
        query="家里既有木地板又有地毯，选扫拖机器人要看什么配置？",
        chunks=[buying_chunk],
        sections=[],
        min_score=0.1,
    )

    assert answer is not None
    assert "低出水量可调" in answer or "拖布抬升" in answer
    assert "地毯增压" in answer or "更强吸力" in answer
    assert "越障" in answer


def test_build_direct_grounded_answer_long_term_storage_uses_storage_steps():
    chunk = Chunk(
        document_id="default/maintenance.txt",
        chunk_id=0,
        content=(
            "## 长期存放维护（20条）\n"
            "1. 存放前，将机器人完全充满电（电量至80%-90%），避免亏电存放导致电池损坏。\n"
            "2. 存放前，全面清理机器人机身、尘盒、滤网、主刷、边刷、拖布等所有配件，无灰尘、污渍残留。\n"
            "3. 扫拖一体机器人存放前，排空水箱、污水仓内的所有水分，用干布擦拭干净，晾干后再存放。\n"
            "6. 存放时，将机器人放在阴凉、干燥、通风的环境。\n"
            "11. 长期存放（＞1个月），每1-2个月给机器人补电一次，将电量充至80%-90%，避免电池亏电。"
        ),
        title="维护保养.txt",
        source_type="file",
    )

    answer = main._build_direct_grounded_answer(
        query="机器人如果要长期存放，应该怎么保养？",
        chunks=[chunk],
        sections=[],
        min_score=0.1,
    )

    assert answer is not None
    assert "清洁机身和配件" in answer
    assert "排空水箱" in answer
    assert "80% 到 90%" in answer
    assert "阴凉干燥" in answer
    assert "补电" in answer


def test_query_specific_keyword_boosts_cover_remaining_eval_cases():
    mop_boosts = main._query_specific_keyword_boosts("水箱里加清洁液应该怎么配比？")
    storage_boosts = main._query_specific_keyword_boosts("机器人如果要长期存放，应该怎么保养？")

    assert any("1:100" in item for item in mop_boosts)
    assert any("长期存放维护" in item for item in storage_boosts)


def test_build_excerpt_for_boosted_query_focuses_cleaning_liquid_ratio():
    content = (
        "1. **扫拖一体机器人可以只扫地不拖地吗？**\n"
        "- 可以。\n"
        "14. **如何给水箱添加清洁液？**\n"
        "- 建议使用机型专用清洁液，按1:100的比例与清水混合后加注，不可直接加浓清洁液。\n"
        "15. **电动拖布不旋转怎么办？**\n"
        "- 检查拖布是否被毛发缠绕。"
    )

    excerpt = main._build_excerpt_for_boosted_query(content, "水箱里加清洁液应该怎么配比？")

    assert excerpt is not None
    assert "如何给水箱添加清洁液" in excerpt
    assert "按1:100的比例" in excerpt
    assert "只扫地不拖地" not in excerpt


def test_focus_chunks_for_query_replaces_long_chunk_with_excerpt():
    long_chunk = Chunk(
        document_id="default/mop.txt",
        chunk_id=0,
        content=(
            "1. **扫拖一体机器人可以只扫地不拖地吗？**\n"
            "- 可以。\n"
            "2. **如何设置先扫后拖的清洁流程？**\n"
            "- 可在APP中设置。\n"
            "14. **如何给水箱添加清洁液？**\n"
            "- 建议使用机型专用清洁液，按1:100的比例与清水混合后加注，不可直接加浓清洁液。\n"
            "15. **电动拖布不旋转怎么办？**\n"
            "- 检查拖布是否被毛发缠绕。"
        ),
        title="扫拖一体机器人100问.txt",
        source_type="file",
    )

    focused = main._focus_chunks_for_query([long_chunk], "水箱里加清洁液应该怎么配比？")

    assert len(focused) == 1
    assert "如何给水箱添加清洁液" in focused[0].content
    assert "按1:100的比例" in focused[0].content
    assert "扫拖一体机器人可以只扫地不拖地" not in focused[0].content

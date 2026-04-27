from pydantic import BaseModel, Field


class Analyze(BaseModel):
    """TODO."""

    prompt: str


class Hallucination(BaseModel):
    """TODO."""

    introduced_new_objects: bool = Field(
        description="指示是否引入了原图中不存在的新物体"
    )
    introduced_unreasonable_textures: bool = Field(
        description="指示是否生成了不合理或奇怪的纹理"
    )
    reasoning: str = Field(description="详细解释是否存在任何幻觉情况及原因")


class QualityMetrics(BaseModel):
    """TODO."""

    is_clearer: bool = Field(description="图像是否变得更加清晰")
    color_standard_met: bool = Field(description="颜色是否符合标准")
    contrast_standard_met: bool = Field(description="对比度是否符合标准")
    texture_standard_met: bool = Field(description="纹理细节是否符合标准")
    reasoning: str = Field(
        description="解释这些质量改进的具体情况以及它们为何符合或不符合标准"
    )


class EvaluationResponse(BaseModel):
    """TODO."""

    hallucination: Hallucination
    quality_metrics: QualityMetrics
    overall_score: int = Field(
        ge=0, le=5, description="综合评分, 0-5 整数"
    )
    final_verdict: str = Field(description="总结整个评估结果并给出最终结论")


class ReflectResponse(BaseModel):
    """Reflect 阶段的输出."""

    reasoning: str = Field(description="分析各分支的权衡推理过程")
    best_branch: str = Field(description="选中的最佳分支名称")
    should_continue: bool = Field(description="是否需要继续下一轮修复")

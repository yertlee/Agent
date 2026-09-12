# R3 project-authored demo corpus v1

这里的政策和商品资料是项目自建演示内容，仅用于 R3 检索、证据链和拒答验收，不代表任何外部平台的当前规则。订单、物流和用户隐私事实不属于本 corpus。

`manifest.json` 冻结 source metadata、时间范围、chunk 规则、模型、RRF 和 checksum。运行时通过 `agent.r3_rag.load_manifest` 校验 source 与 manifest checksum。

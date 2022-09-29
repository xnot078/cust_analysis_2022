# %% 目標
# 幫每個客戶每年標記轉移的標籤
#
```mermaid
全體 -> 有轉移(label=2, link=f'{ref}->{landing}')
全體 -> 沒轉移
沒轉移 -> 轉移到同通路(label=1, link=f'{ref}->{landing=ref}')
沒轉移 -> 流失(label=0, link=f'{ref}->loss')
```

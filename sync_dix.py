import os
import requests
import cloudscraper

def sync_data():
    url = "https://squeezemetrics.com/api/dix.csv"
    print("⏳ Sentinel Data Pipeline: 开始尝试同步暗池数据...")
    
    try:
        # 实例化高级解密爬虫
        scraper = cloudscraper.create_scraper(
            browser={
                'browser': 'chrome',
                'platform': 'windows',
                'desktop': True
            }
        )
        
        # 设定较长超时，应对跨国网络波动
        response = scraper.get(url, timeout=30)
        
        if response.status_code == 200 and len(response.text) > 1000:
            # 成功抓取，直接重写本地的 dix.csv
            with open("dix.csv", "w", encoding="utf-8") as f:
                f.write(response.text)
            print("✅ 成功！最新 dix.csv 已安全下载至本地仓库。")
            return True
        else:
            print(f"❌ 获取失败。状态码: {response.status_code}，数据长度异常。")
            return False
            
    except Exception as e:
        print(f"💥 触发异常: {e}")
        return False

if __name__ == "__main__":
    sync_data()

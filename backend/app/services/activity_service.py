import datetime
from typing import List, Dict

class ActivityService:
    def __init__(self):
        # 存储最近的系统活动
        self.activities = [
            {
                'id': 1,
                'content': '系统初始化完成',
                'type': 'info',
                'timestamp': datetime.datetime.now().isoformat()
            }
        ]
        self.max_activities = 20

    def add_activity(self, content: str, activity_type: str = 'info'):
        activity = {
            'id': len(self.activities) + 1,
            'content': content,
            'type': activity_type,
            'timestamp': datetime.datetime.now().isoformat()
        }
        self.activities.insert(0, activity) # 最新的放在前面
        
        # 保持列表长度
        if len(self.activities) > self.max_activities:
            self.activities = self.activities[:self.max_activities]
            
    def get_activities(self) -> List[Dict]:
        return self.activities

# 全局单例
activity_service = ActivityService()

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
        self.next_activity_id = 2
        self.max_activities = 20

    def add_activity(self, content: str, activity_type: str = 'info', metadata: Dict = None):
        activity = {
            'id': self.next_activity_id,
            'content': content,
            'type': activity_type,
            'timestamp': datetime.datetime.now().isoformat()
        }
        if metadata:
            activity['metadata'] = metadata
        self.next_activity_id += 1
        self.activities.insert(0, activity) # 最新的放在前面
        
        # 保持列表长度
        if len(self.activities) > self.max_activities:
            self.activities = self.activities[:self.max_activities]
            
    def get_activities(self) -> List[Dict]:
        return sorted(self.activities, key=lambda activity: activity.get('timestamp', ''), reverse=True)

# 全局单例
activity_service = ActivityService()

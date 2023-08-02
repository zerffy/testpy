from zhdate import ZhDate

from datetime import datetime
d2 = datetime(1999, 3, 19)
date2 = ZhDate.from_datetime(d2)
# 输出结果
print(date2)
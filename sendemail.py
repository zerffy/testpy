import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import time

def gettime():
    time1 = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return time1

def send_email(sender_email, sender_password, receiver_email, subject, message):
    # 邮件服务器的地址和端口
    smtp_server = 'smtp.163.com'
    smtp_port = 25

    # 创建包含邮件内容的多部分（multipart）消息
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # 添加纯文本消息
    msg.attach(MIMEText(message, 'plain'))

    try:
        # 建立SMTP连接
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)

        # 发送邮件
        server.send_message(msg)
        print('邮件发送成功！')

    except Exception as e:
        print('邮件发送失败:', str(e))

    finally:
        # 关闭SMTP连接
        server.quit()

# 发件人和收件人的邮箱地址
sender_email = 'bluemrl@163.com'
sender_password = 'CTHXDYKHKNYRQKGB'
receiver_email = '928749695@qq.com'

# 邮件主题和内容
subject = '测试邮件'
message = '这是一封测试邮件。\n' + '发送时间：' + gettime()

# 调用发送邮件函数
send_email(sender_email, sender_password, receiver_email, subject, message)

import socket
import time
import math

class AIRobot():
    def __init__(self, key, name="CurlingAI", show_msg=False):
        #服务器IP(固定不变无需修改)
        host = '192.168.0.127'
        #连接端口(固定不变无需修改)
        port = 7788

        #新建Socket对象
        self.ai_sock = socket.socket()
        #创建Socket连接
        self.ai_sock.connect((host,port))
        print("已建立socket连接", host, port)
        
        #是否显示接收/发送的消息
        self.show_msg = show_msg
        #发送连接密钥
        self.send_msg("CONNECTKEY:" + key)

        #设定机器人名称
        self.name = name
        #初始化冰壶位置
        self.position = [0]*32
        #初始化冰壶运动信息
        self.motioninfo = [0]*5
        #设定起始局数
        self.round_num = 0

    #通过socket对象发送消息
    def send_msg(self, msg):
        if (self.show_msg):
            print("  >>>> " + msg)
        #将消息数据从字符串类型转换为bytes类型后发送
        self.ai_sock.send(msg.strip().encode())

    #通过socket对象接收消息并进行解析
    def recv_msg(self):
        #为避免TCP粘包问题，数字冰壶服务器发送给AI选手的每一条信息均以0（数值为0的字节）结尾
        #这里采用了逐个字节接收后拼接的方式处理信息，多条信息之间以0为信息终结符
        buffer = bytearray()
        while True:
            #接收1个字节
            data = self.ai_sock.recv(1)
            #接收到空数据或者信息处终结符(0)即中断循环
            if not data or data == b'\0':
                break
            #将当前字节拼接到缓存中
            buffer.extend(data)
        #将消息数据从bytes类型转换为字符串类型后去除前后空格
        msg_str = buffer.decode().strip()
        if (self.show_msg):
            print("<<<< " + msg_str)

        #用空格将消息字符串分隔为列表
        msg_list = msg_str.split(" ")
        #列表中第一个项为消息代码
        msg_code = msg_list[0]
        #列表中后续的项为各个参数
        msg_list.pop(0)
        #返回消息代码和消息参数列表
        return msg_code, msg_list

    #与大本营中心距离
    def get_dist(self, x, y):
        House_x = 2.375
        House_y = 4.88
        return math.sqrt((x-House_x)**2+(y-House_y)**2)

    #大本营内是否有壶
    def is_in_house(self, dist):
        House_R = 1.830
        Stone_R = 0.145
        if dist<(House_R+Stone_R):
            return 1
        else:
            return 0
        
    def recv_setstate(self, msg_list):
        #当前完成投掷数
        self.shot_num = int(msg_list[0])
        #当前完成对局数
        self.round_num = int(msg_list[1])
        #总对局数
        self.round_total = int(msg_list[2])
        #预备投掷者（0为持蓝色球者，1为持红色球者）
        self.next_shot = int(msg_list[3])
 
    #基础AI策略
    def get_bestshot(self):
        if (self.show_msg):
            print("============第%d局第%d壶============" % (self.round_num+1, self.shot_num+1))
        #初始化先手壶和后手壶的坐标列表
        init_x, init_y, gote_x, gote_y = [0]*8, [0]*8, [0]*8, [0]*8
        #初始化大本营中冰壶球信息列表
        stone_in_house = []
        #获取大本营中冰壶球信息
        for n in range(8):
            stone_is_init = True
            init_x[n], init_y[n] = float(self.position[n*4]), float(self.position[n*4+1])
            gote_x[n], gote_y[n] = float(self.position[n*4+2]), float(self.position[n*4+3])
            for (x, y) in [(init_x[n], init_y[n]), (gote_x[n], gote_y[n])]:
                distance = self.get_dist(x, y)
                if self.is_in_house(distance):
                    stone_in_house.append([distance, x, y, stone_is_init])
                stone_is_init = False

        #大本营内没有球，向大本营中心打球
        if len(stone_in_house) == 0:
            shot_msg = "BESTSHOT 3.0 0 0"
        #大本营有球
        else:
            stone_in_house=sorted(stone_in_house)
            _, x, y, stone_is_init = stone_in_house[0]
            #离大本营中心最近的球是己方的，保护
            if self.player_is_init == stone_is_init:
                v0 = 3.613 - 0.12234*y - 0.3
                h0 = x - 2.375
            #离大本营中心最近的球是对方的，击飞
            else:
                v0 = 3.613 - 0.12234*y + 1
                h0 = x - 2.375
            shot_msg = "BESTSHOT " + str(v0) + " " + str(h0) + " 0"
        return shot_msg

    #接收并处理消息
    def recv_forever(self):
        #空消息计数器归零
        retNullTime = 0
        self.on_line = True
        
        while(self.on_line):
            #接收消息并解析
            msg_code, msg_list = self.recv_msg()
            #如果接到空消息则将计数器加一
            if msg_code == "":
                retNullTime = retNullTime + 1
            #如果接到五条空消息则关闭Socket连接
            if retNullTime == 5:
                break    
            #如果消息代码是……
            if msg_code == "CONNECTNAME":
                if msg_list[0] == "Player1":
                    self.player_is_init = True
                    print("玩家1，首局先手")
                else:
                    self.player_is_init = False
                    print("玩家2，首局后手")
            if msg_code == "ISREADY":
                #发送"READYOK"
                self.send_msg("READYOK")
                time.sleep(0.5)
                #发送"NAME"和AI选手名
                self.send_msg("NAME " + self.name)    
                print(self.name +" 准备完毕！")
            if msg_code == "NEWGAME":
                time0 = time.time()
            if msg_code=="SETSTATE":
                self.recv_setstate(msg_list)
            if msg_code=="POSITION":
                for n in range(32):
                    self.position[n] = float(msg_list[n])
            if msg_code == "GO":
                #制定策略生成投壶信息
                shot_msg = self.get_bestshot()
                #发送投壶消息
                self.send_msg(shot_msg)
            if msg_code == "MOTIONINFO":
                for n in range(5):
                    self.motioninfo[n] = float(msg_list[n])
            #如果消息代码是"SCORE"
            if msg_code == "SCORE":
                time1 = time.time()
                print("%s 第%d局耗时%.1f秒" % (time.strftime("[%Y/%m/%d %H:%M:%S]"),self.round_num+1,time1-time0), end=" ")
                time0 = time1
                #从消息参数列表中获取得分
                self.score = int(msg_list[0])
                #得分的队伍在下一局是先手
                if  self.score > 0:
                    print("我方得"+str(self.score)+"分", end=" ")
                    #如果不是无限对战模式(固定先后手)
                    if self.round_total != (-1):
                        self.player_is_init = True
                #失分的队伍在下一局是后手
                elif self.score < 0:
                    print("对方得"+str(self.score*-1)+"分", end=" ")
                    #如果不是无限对战模式(固定先后手)
                    if self.round_total != (-1):
                        self.player_is_init = False
                #平局下一局交换先后手
                else:
                    print("双方均未得分", end=" ")
                    #如果不是无限对战模式(固定先后手)
                    if self.round_total != (-1):
                        self.player_is_init = not self.player_is_init
                if (self.player_is_init):
                    print("我方下局先手")
                else:
                    print("我方下局后手")
            #如果消息代码是"GAMEOVER"
            if msg_code == "GAMEOVER":
                if  msg_list[0] == "WIN":
                    print("我方获胜")
                elif msg_list[0] == "LOSE":
                    print("对方获胜")
                else:
                    print("双方平局")

        #关闭Socket连接
        self.ai_sock.close()
        print("已关闭socket连接")
        
if __name__ == '__main__':
    #根据数字冰壶服务器界面中给出的连接信息修改CONNECTKEY，注意这个数据每次启动都会改变。
    key = "test2023_2_1a5e475b-6a1c-499e-ae43-b67d72f2ab69"
    #初始化AI选手
    airobot = AIRobot(key)
    #启动AI选手处理和服务器的通讯
    airobot.recv_forever()
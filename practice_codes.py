# import yaml

# with open('config.yaml') as f:
#     data = yaml.load(f, Loader=yaml.FullLoader)
#     episodes = data['EPISODES']
#     print(episodes)

class A():
    def __init__(self):
        pass
    def run(self, b):
        return b+2

a = A()
print(a.run(2))
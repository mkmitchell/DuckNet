import queue

class PubSub:
    '''Global publish-subscribe class for push messages to the UI'''
    subscribers = []

    @classmethod
    def subscribe(cls):
        q = queue.Queue(maxsize=5)
        cls.subscribers.append(q)
        return q

    @classmethod
    def publish(cls, msg, event='message'):
        for i in reversed(range(len(cls.subscribers))):
            try:
                cls.subscribers[i].put_nowait((event, msg))
            except queue.Full:
                del cls.subscribers[i]

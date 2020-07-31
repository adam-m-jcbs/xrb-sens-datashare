import concurrent.futures
from multiprocessing import Queue
import time


def qwait(q, id):
    return id, q.get()

def qput(q, delay, message):
    time.sleep(delay)
    q.put(message)


def run():
    nq = 5

    px = concurrent.futures.ThreadPoolExecutor()

    qq = [Queue() for i in range(nq)]

    futures = [px.submit(qwait, qq[i], i) for i in range(nq)]

    px.submit(qput, qq[2], 1, 'A')
    px.submit(qput, qq[2], 2, 'B')
    px.submit(qput, qq[1], 1, 'C')

    print('running stuff')

    while True:
        done, not_sone = concurrent.futures.wait(
            futures,
            timeout = 2,
            return_when=concurrent.futures.FIRST_COMPLETED)

        for d in done:
            id, data = d.result()
            print(id, data)
            futures.remove(d)
            futures.append(px.submit(qwait, qq[id], id))

            if len(done) == 0:
                for f in futures:
                    loop.call_soon(f.cancel)

        if len(done) == 0:
            for f in futures:
                print(f.cancel())
            break

    print('---')
    for f in futures:
        print(f.cancelled())
    px.shutdown(wait = False)

import asyncio

async def aput(q, delay, message):
    await asyncio.sleep(delay)
    q.put(message)

async def main(futures, loop, qq):
    n = len(futures)
    while True:
        done, not_sone = await asyncio.wait(
            futures,
            timeout = 2,
            return_when=concurrent.futures.FIRST_COMPLETED)

        for d in done:
            id, data = d.result()
            print(id, data)
            futures.remove(d)
            futures.append(loop.run_in_executor(None, qwait, qq[id], id))

        if len(done) == 0:
            for f in futures:
                loop.call_soon(f.cancel)
            break
    await asyncio.wait(
        futures,
        return_when=concurrent.futures.ALL_COMPLETED)

def run2():

    nq = 5

    qq = [Queue() for i in range(nq)]

    qq[0].put('X')

    loop = asyncio.get_event_loop()
    if loop.is_closed():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    px = concurrent.futures.ThreadPoolExecutor()

    futures = [loop.run_in_executor(px, qwait, qq[i], i) for i in range(nq)]

    asyncio.ensure_future(aput(qq[2], 1, 'A'))
    asyncio.ensure_future(aput(qq[2], 2, 'B'))
    asyncio.ensure_future(aput(qq[1], 1, 'C'))

    loop.call_later(1.5, qq[3].put, 'D')

    print('running stuff')

    loop.run_until_complete(main(futures, loop, qq))

    loop.close()
    px.shutdown(wait = False)

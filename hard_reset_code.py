# This code resets the controller and clears any error messages and exists from motor braking

import asyncio
import moteus

async def main():
    c = moteus.Controller(id=1)
    s = moteus.Stream(c)

    await s.write_message(b'tel stop')
    await s.flush_read()
    await s.command(b'd stop')


if __name__ == '__main__':
    asyncio.run(main())

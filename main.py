"""
Demonstration of problems when NOT using async context managers properly.
Run this to see resource leaks, warnings, and other issues in action.
"""

import asyncio
import httpx
import gc
import warnings
import psutil
import os
import sys
from typing import List


# Enable all warnings to see ResourceWarnings
warnings.simplefilter("always", ResourceWarning)


class BadAsyncHTTPClient:
    """Example of WRONG implementation - no context manager."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url, timeout=5.0)

    async def get(self, url: str):
        return await self.client.get(url)

    # No __aenter__, __aexit__, no cleanup method!


class ForgetfulAsyncHTTPClient:
    """Implementation where user forgets to close."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=5.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.aclose()

    async def get(self, url: str):
        return await self.client.get(url)


class ProperAsyncHTTPClient:
    """CORRECT implementation with proper cleanup."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self.client = None

    async def __aenter__(self):
        self.client = httpx.AsyncClient(base_url=self.base_url, timeout=5.0)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()

    async def get(self, url: str):
        return await self.client.get(url)


def get_process_info():
    """Get current process resource usage."""
    process = psutil.Process(os.getpid())
    return {
        'fds': process.num_fds() if hasattr(process, 'num_fds') else len(process.open_files()),
        'threads': process.num_threads(),
        'memory_mb': process.memory_info().rss / 1024 / 1024,
        'connections': len(process.connections()),
    }


def print_resources(label: str, info: dict):
    """Pretty print resource usage."""
    print(f"\n{label}:")
    print(f"  File Descriptors: {info['fds']}")
    print(f"  Threads: {info['threads']}")
    print(f"  Memory: {info['memory_mb']:.2f} MB")
    print(f"  Network Connections: {info['connections']}")


async def demo_1_no_cleanup():
    """PROBLEM 1: Creating clients without any cleanup."""
    print("\n" + "="*70)
    print("DEMO 1: No Cleanup - Creating clients without closing them")
    print("="*70)

    initial = get_process_info()
    print_resources("Initial state", initial)

    # Create multiple clients and never close them
    clients: List[BadAsyncHTTPClient] = []
    for i in range(10):
        client = BadAsyncHTTPClient(f"https://api.example{i}.com")
        clients.append(client)
        print(f"  Created client {i+1}/10...")

    during = get_process_info()
    print_resources("After creating 10 clients (not closed)", during)

    # Even if we delete references, without proper cleanup...
    clients.clear()
    gc.collect()
    await asyncio.sleep(0.2)

    after_gc = get_process_info()
    print_resources("After del + garbage collection", after_gc)

    print("\n⚠️  PROBLEM: File descriptors and connections leaked!")
    print(f"   FD leak: {after_gc['fds'] - initial['fds']} descriptors")
    print(f"   Connection leak: {after_gc['connections'] - initial['connections']} connections")
    print("   These will stay open until the program exits!")


async def demo_2_forgot_context_manager():
    """PROBLEM 2: Has context manager but user forgets to use it."""
    print("\n" + "="*70)
    print("DEMO 2: Forgot to use 'async with' - Manual usage without cleanup")
    print("="*70)

    initial = get_process_info()
    print_resources("Initial state", initial)

    # User creates client but forgets to use 'async with'
    print("\n  Creating client without 'async with'...")
    client = ForgetfulAsyncHTTPClient("https://httpbin.org")

    # Manually entering context but forgetting to exit
    await client.__aenter__()

    # Do some work
    print("  Client is active, connections open...")

    during = get_process_info()
    print_resources("While client is active", during)

    # Oops! User forgets to call __aexit__ or aclose()
    # Just deletes the reference
    del client
    gc.collect()
    await asyncio.sleep(0.2)

    after = get_process_info()
    print_resources("After deleting client (no cleanup)", after)

    print("\n⚠️  PROBLEM: Resources not cleaned up!")
    print("   User forgot to exit context manager or call close()")


async def demo_3_exception_without_cleanup():
    """PROBLEM 3: Exception occurs and cleanup code is skipped."""
    print("\n" + "="*70)
    print("DEMO 3: Exception Without Proper Cleanup")
    print("="*70)

    initial = get_process_info()
    print_resources("Initial state", initial)

    try:
        print("\n  Creating client and simulating work...")
        client = BadAsyncHTTPClient("https://httpbin.org")

        during = get_process_info()
        print_resources("Client created", during)

        # Simulate some work then an exception
        print("  Working... then exception occurs!")
        raise ValueError("Something went wrong!")

        # This cleanup code never runs!
        await client.client.aclose()  # Never reached

    except ValueError as e:
        print(f"  Caught exception: {e}")
        print("  Cleanup code after exception was NEVER reached!")

    gc.collect()
    await asyncio.sleep(0.2)

    after = get_process_info()
    print_resources("After exception (no cleanup)", after)

    print("\n⚠️  PROBLEM: Exception prevented cleanup!")
    print("   Connection still open because cleanup was after the exception")


async def demo_4_resource_warnings():
    """PROBLEM 4: Python's ResourceWarning for unclosed resources."""
    print("\n" + "="*70)
    print("DEMO 4: ResourceWarning - Python detects unclosed resources")
    print("="*70)

    print("\n  Creating httpx.AsyncClient without closing...")
    print("  Watch for ResourceWarning when garbage collected:\n")

    # Create client and let it go out of scope without closing
    def create_and_abandon():
        client = httpx.AsyncClient(base_url="https://httpbin.org", timeout=5.0)
        # Oops, never closed!
        return None

    create_and_abandon()

    # Force garbage collection to trigger warning
    print("  Forcing garbage collection...")
    gc.collect()
    await asyncio.sleep(0.1)

    print("\n⚠️  PROBLEM: ResourceWarning raised!")
    print("   Python detected unclosed httpx.AsyncClient")
    print("   In production, these warnings indicate resource leaks")


async def demo_5_connection_pool_exhaustion():
    """PROBLEM 5: Eventually exhaust system resources."""
    print("\n" + "="*70)
    print("DEMO 5: Connection Pool Exhaustion")
    print("="*70)

    initial = get_process_info()
    print_resources("Initial state", initial)

    print("\n  Creating 50 clients without cleanup...")
    print("  (This simulates what happens over time in production)\n")

    clients = []
    for i in range(50):
        client = httpx.AsyncClient(base_url=f"https://api{i}.example.com", timeout=5.0)
        clients.append(client)

        if i % 10 == 9:
            current = get_process_info()
            print(f"  After {i+1} clients: {current['fds']} FDs, {current['connections']} connections")

    final = get_process_info()
    print_resources("\nAfter creating 50 unclosed clients", final)

    print("\n⚠️  PROBLEM: System resources exhausted!")
    print(f"   File descriptor increase: {final['fds'] - initial['fds']}")
    print(f"   Connection increase: {final['connections'] - initial['connections']}")
    print("   Eventually you'll hit OS limits (ulimit) and get errors:")
    print("   - 'Too many open files'")
    print("   - 'Cannot create new connection'")
    print("   - Application crashes or hangs")

    # Cleanup for this demo
    print("\n  Cleaning up demo resources...")
    for client in clients:
        await client.aclose()


async def demo_6_proper_usage():
    """Show the CORRECT way with proper cleanup."""
    print("\n" + "="*70)
    print("DEMO 6: CORRECT Usage - Proper async context management")
    print("="*70)

    initial = get_process_info()
    print_resources("Initial state", initial)

    print("\n  Creating 10 clients WITH proper 'async with' cleanup...\n")

    for i in range(10):
        async with ProperAsyncHTTPClient(f"https://api{i}.example.com") as client:
            # Client is open and usable here
            pass
        # Client is automatically closed here!

        if i % 3 == 2:
            current = get_process_info()
            print(f"  After {i+1} clients (properly closed): {current['fds']} FDs")

    gc.collect()
    await asyncio.sleep(0.2)

    final = get_process_info()
    print_resources("\nAfter 10 properly managed clients", final)

    print("\n✅ SUCCESS: Resources properly cleaned up!")
    print(f"   FD delta: {final['fds'] - initial['fds']} (should be ~0)")
    print(f"   Connection delta: {final['connections'] - initial['connections']} (should be 0)")
    print("   All resources returned to the system!")


async def demo_7_concurrent_tasks_leak():
    """PROBLEM 7: Concurrent tasks creating clients without cleanup."""
    print("\n" + "="*70)
    print("DEMO 7: Concurrent Tasks Without Cleanup")
    print("="*70)

    initial = get_process_info()
    print_resources("Initial state", initial)

    async def bad_worker(worker_id: int):
        """Worker that creates client but doesn't clean up."""
        client = httpx.AsyncClient(base_url=f"https://worker{worker_id}.example.com", timeout=5.0)
        await asyncio.sleep(0.1)
        # Oops! Forgot to close client
        return worker_id

    print("\n  Starting 20 concurrent workers (no cleanup)...")
    results = await asyncio.gather(*[bad_worker(i) for i in range(20)])
    print(f"  All {len(results)} workers completed")

    gc.collect()
    await asyncio.sleep(0.2)

    after = get_process_info()
    print_resources("\nAfter concurrent workers finished", after)

    print("\n⚠️  PROBLEM: Each worker leaked resources!")
    print(f"   FD leak: {after['fds'] - initial['fds']}")
    print("   In a web server with many requests, this quickly becomes critical!")


async def demo_8_forgotten_await():
    """PROBLEM 8: Forgetting to await cleanup methods."""
    print("\n" + "="*70)
    print("DEMO 8: Forgot to 'await' cleanup - Silent failure")
    print("="*70)

    initial = get_process_info()
    print_resources("Initial state", initial)

    print("\n  Creating client and 'closing' without await...")

    client = httpx.AsyncClient(base_url="https://httpbin.org", timeout=5.0)

    # WRONG: Forgot to await!
    client.aclose()  # Returns a coroutine but doesn't execute!

    print("  Called client.aclose() but forgot 'await'")
    print("  Python may warn: 'coroutine was never awaited'\n")

    gc.collect()
    await asyncio.sleep(0.2)

    after = get_process_info()
    print_resources("After 'closing' without await", after)

    print("\n⚠️  PROBLEM: Close didn't actually execute!")
    print("   Without 'await', the cleanup code never ran")
    print("   Resources still open!")

    # Actually close it
    await client.aclose()


async def demo_11_exception_in_get_without_context():
    """PROBLEM 11: Using .get() without entering context manager, exception occurs."""
    print("\n" + "="*70)
    print("DEMO 11: Exception Inside .get() - No Context Manager Used At All")
    print("="*70)

    print("\nWhat happens when:")
    print("  1. You create a client but DON'T use 'async with'")
    print("  2. You call .get() directly")
    print("  3. An exception occurs INSIDE the .get() method")
    print("  4. You never manually clean up\n")

    initial = get_process_info()
    print_resources("Initial state", initial)

    # Create client but DON'T enter context manager
    print("\n  Creating ForgetfulAsyncHTTPClient...")
    client = ForgetfulAsyncHTTPClient("https://httpbin.org")

    print("  NOT using 'async with' - just calling .get() directly...")

    try:
        # Try to call .get() without ever calling __aenter__
        print("  Calling client.get()...")
        response = await client.get("/get")
        print(f"  Response: {response.status_code}")

    except AttributeError as e:
        print(f"  ❌ AttributeError: {e}")
        print("  💥 client.client is None! We never called __aenter__!")
        print("  The .get() method tried to use self.client which doesn't exist")
    except Exception as e:
        print(f"  ❌ Exception: {type(e).__name__}: {e}")

    print("\n  Now let's try the REAL scenario you asked about:")
    print("  What if we manually create the httpx client but exception occurs?\n")

    # Manually set up the client (simulating someone who bypasses context manager)
    client2 = ForgetfulAsyncHTTPClient("https://httpbin.org")
    client2.client = httpx.AsyncClient(base_url="https://httpbin.org", timeout=5.0)

    during = get_process_info()
    print_resources("After manually creating httpx client", during)

    print("\n  Client is now active with open connection...")
    print("  Calling .get() with an invalid URL that will cause exception...\n")

    try:
        # This will cause an exception (timeout, DNS error, etc.)
        await client2.get("https://this-domain-definitely-does-not-exist-12345.com/api")

    except Exception as e:
        print(f"  ❌ Exception during .get(): {type(e).__name__}")
        print(f"     {str(e)[:100]}...")
        print("\n  🔍 What happened to the httpx.AsyncClient?")
        print(f"     client2.client exists: {client2.client is not None}")
        print(f"     client2.client.is_closed: {client2.client.is_closed}")
        print("     ⚠️  The httpx client is still OPEN!")
        print("     The exception didn't close it - connections are leaked!")

    # Don't clean up - see what happens
    gc.collect()
    await asyncio.sleep(0.2)

    after = get_process_info()
    print_resources("\nAfter exception (no cleanup)", after)

    print("\n⚠️  CRITICAL PROBLEM:")
    print(f"   FD leak: {after['fds'] - initial['fds']} file descriptors")
    print(f"   Connection leak: {after['connections'] - initial['connections']} connections")
    print("   The exception in .get() did NOT close the underlying client!")
    print("   Resources are permanently leaked!")

    # Now show what happens with proper async with
    print("\n\n--- Now with proper 'async with' ---")

    initial2 = get_process_info()
    print_resources("Initial state", initial2)

    try:
        async with ForgetfulAsyncHTTPClient("https://httpbin.org") as client3:
            during2 = get_process_info()
            print_resources("Inside 'async with'", during2)

            print("\n  Calling .get() with invalid URL...")
            await client3.get("https://this-domain-definitely-does-not-exist-12345.com/api")

    except Exception as e:
        print(f"  ❌ Exception: {type(e).__name__}")
        print("  ✅ BUT: __aexit__ was called automatically!")

    gc.collect()
    await asyncio.sleep(0.2)

    after2 = get_process_info()
    print_resources("\nAfter exception (with async with)", after2)

    print("\n✅ SUCCESS with 'async with':")
    print(f"   FD delta: {after2['fds'] - initial2['fds']} (cleaned up!)")
    print(f"   Connection delta: {after2['connections'] - initial2['connections']} (cleaned up!)")

    print("\n🔑 KEY INSIGHTS:")
    print("   1. Exception in .get() does NOT automatically close the client")
    print("   2. httpx.AsyncClient stays open even after exception")
    print("   3. Without 'async with', YOU must handle cleanup in try/finally")
    print("   4. With 'async with', cleanup happens automatically")
    print("\n   Bottom line: Exception in .get() → client still open → leak!")


async def demo_12_multiple_get_calls_with_exception():
    """PROBLEM 12: Multiple .get() calls, exception on one of them."""
    print("\n" + "="*70)
    print("DEMO 12: Multiple .get() Calls - Exception Partway Through")
    print("="*70)

    print("\nRealistic scenario: Making multiple API calls in sequence")
    print("What if the 3rd call fails?\n")

    initial = get_process_info()
    print_resources("Initial state", initial)

    # WITHOUT async with
    print("\n--- WITHOUT 'async with' ---")

    client = ForgetfulAsyncHTTPClient("https://httpbin.org")
    # Manually create the httpx client (bypassing context manager)
    client.client = httpx.AsyncClient(base_url="https://httpbin.org", timeout=5.0)

    print("  Client created, making multiple .get() calls...")

    try:
        # Call 1 - succeeds (we'll mock it)
        print("  Call 1: Success (simulated)")

        # Call 2 - succeeds (we'll mock it)
        print("  Call 2: Success (simulated)")

        # Call 3 - FAILS
        print("  Call 3: Attempting request that will fail...")
        await client.get("https://invalid-domain-12345.com/api")

        # These never execute
        print("  Call 4: Never reached")
        print("  Cleanup: Never reached")
        await client.client.aclose()

    except Exception as e:
        print(f"  ❌ Exception on call 3: {type(e).__name__}")
        print(f"     client.client.is_closed: {client.client.is_closed}")
        print("     ⚠️  Cleanup code was never reached!")

    after_bad = get_process_info()
    print_resources("\nAfter exception (no async with)", after_bad)

    print(f"\n⚠️  Leaked: {after_bad['fds'] - initial['fds']} FDs, {after_bad['connections'] - initial['connections']} connections")

    # WITH async with
    print("\n\n--- WITH 'async with' ---")

    initial2 = get_process_info()

    try:
        async with ForgetfulAsyncHTTPClient("https://httpbin.org") as client2:
            print("  Client created with 'async with', making calls...")

            print("  Call 1: Success (simulated)")
            print("  Call 2: Success (simulated)")
            print("  Call 3: Attempting request that will fail...")

            await client2.get("https://invalid-domain-12345.com/api")

            # These never execute
            print("  Call 4: Never reached")

    except Exception as e:
        print(f"  ❌ Exception on call 3: {type(e).__name__}")
        print("  ✅ But __aexit__ still ran - client cleaned up!")

    gc.collect()
    await asyncio.sleep(0.2)

    after_good = get_process_info()
    print_resources("\nAfter exception (with async with)", after_good)

    print(f"\n✅ No leak: {after_good['fds'] - initial2['fds']} FD delta")

    print("\n💡 REAL WORLD IMPACT:")
    print("   Imagine a microservice making 10 API calls per request")
    print("   If call #7 fails occasionally:")
    print("   - WITHOUT 'async with': Leak on every failure")
    print("   - WITH 'async with': No leak ever")
    print("   After 1000 failed requests: 1000 leaked connections vs 0")


async def demo_13_thread_behavior_on_exception():
    """PROBLEM 13: What happens to background threads/tasks on exception?"""
    print("\n" + "="*70)
    print("DEMO 13: Background Threads/Tasks When Exception Occurs")
    print("="*70)

    print("\nhttpx.AsyncClient may spawn background tasks for:")
    print("  - Connection pooling")
    print("  - Keep-alive pings")
    print("  - Timeout monitoring")
    print("\nWhat happens to these when .get() raises an exception?\n")

    import threading

    initial = get_process_info()
    initial_threads = threading.active_count()
    initial_tasks = len([t for t in asyncio.all_tasks() if not t.done()])

    print_resources("Initial state", initial)
    print(f"  Active threads: {initial_threads}")
    print(f"  Active asyncio tasks: {initial_tasks}")

    # Create client without context manager
    print("\n  Creating client without 'async with'...")
    client = ForgetfulAsyncHTTPClient("https://httpbin.org")
    client.client = httpx.AsyncClient(
        base_url="https://httpbin.org",
        timeout=5.0,
        limits=httpx.Limits(max_connections=10)
    )

    during = get_process_info()
    during_threads = threading.active_count()
    during_tasks = len([t for t in asyncio.all_tasks() if not t.done()])

    print_resources("\nAfter creating client", during)
    print(f"  Active threads: {during_threads} (delta: +{during_threads - initial_threads})")
    print(f"  Active asyncio tasks: {during_tasks} (delta: +{during_tasks - initial_tasks})")

    print("\n  Attempting .get() that will cause exception...")

    try:
        await client.get("https://invalid-domain-that-does-not-exist-12345.com/test")
    except Exception as e:
        print(f"  ❌ Exception: {type(e).__name__}")

    # Check state after exception
    await asyncio.sleep(0.3)  # Give time for any cleanup

    after = get_process_info()
    after_threads = threading.active_count()
    after_tasks = len([t for t in asyncio.all_tasks() if not t.done()])

    print_resources("\nAfter exception (no cleanup)", after)
    print(f"  Active threads: {after_threads} (delta: +{after_threads - initial_threads})")
    print(f"  Active asyncio tasks: {after_tasks} (delta: +{after_tasks - initial_tasks})")
    print(f"  client.client.is_closed: {client.client.is_closed}")

    print("\n⚠️  PROBLEM:")
    print("   The httpx client is still OPEN")
    print("   Background resources may still be active")
    print("   They won't be cleaned up until:")
    print("   - You explicitly call aclose()")
    print("   - The program exits")
    print("   - Python's garbage collector finalizes the object (unreliable)")

    print("\n  Explicitly closing now to compare...")
    await client.client.aclose()
    await asyncio.sleep(0.2)

    after_close = get_process_info()
    after_close_threads = threading.active_count()

    print_resources("\nAfter explicit close", after_close)
    print(f"  Active threads: {after_close_threads}")
    print(f"  Thread cleanup: {after_threads - after_close_threads} threads stopped")

    print("\n✅ Only after explicit close are resources freed!")


async def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("ASYNC CONTEXT MANAGER FAILURE DEMONSTRATIONS")
    print("="*70)
    print("\nThis script demonstrates what goes wrong without proper")
    print("async context management in Python.\n")

    try:
        await demo_1_no_cleanup()
        await asyncio.sleep(1)

        await demo_2_forgot_context_manager()
        await asyncio.sleep(1)

        await demo_3_exception_without_cleanup()
        await asyncio.sleep(1)

        await demo_4_resource_warnings()
        await asyncio.sleep(1)

        await demo_5_connection_pool_exhaustion()
        await asyncio.sleep(1)

        await demo_6_proper_usage()
        await asyncio.sleep(1)

        await demo_7_concurrent_tasks_leak()
        await asyncio.sleep(1)

        await demo_8_forgotten_await()
        await asyncio.sleep(1)

        await demo_11_exception_in_get_without_context()  # NEW
        await asyncio.sleep(1)

        await demo_12_multiple_get_calls_with_exception()  # NEW
        await asyncio.sleep(1)

        await demo_13_thread_behavior_on_exception()  # NEW

    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

    print("\n" + "="*70)
    print("SUMMARY OF PROBLEMS WITHOUT PROPER async with:")
    print("="*70)
    print("""
1. ❌ File descriptor leaks - eventually hit OS limits
2. ❌ Network connection leaks - ports stay bound
3. ❌ Memory leaks - objects not garbage collected
4. ❌ ResourceWarning from Python - indicators of bugs
5. ❌ Exception handling breaks cleanup - resources stuck open
6. ❌ Thread pool threads not cleaned up
7. ❌ Concurrent tasks multiply the problem
8. ❌ Silent failures when forgetting 'await'

✅ SOLUTION: Always use 'async with' for proper cleanup!

    async with AsyncHTTPClient(...) as client:
        # Use client
        pass
    # Automatically cleaned up here, even on exceptions!
""")


if __name__ == "__main__":
    print("Python version:", sys.version)
    print("Starting demonstrations...\n")
    asyncio.run(main())

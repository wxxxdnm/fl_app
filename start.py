import os
import subprocess
import sys
import platform
import time
import threading
import importlib.util
import argparse

# --- Configuration ---
BACKEND_DIR = "backend"
FRONTEND_DIR = "frontend"
BACKEND_REQUIREMENTS = os.path.join(BACKEND_DIR, "requirements.txt")
FRONTEND_PACKAGE_JSON = os.path.join(FRONTEND_DIR, "package.json")

# --- Helper Functions ---
def run_command(cmd, cwd=None, shell=False, check=True, capture_output=False, timeout=None):
    """Helper to run shell commands."""
    print(f"Executing command: {' '.join(cmd) if isinstance(cmd, list) else cmd} in {cwd if cwd else os.getcwd()}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            shell=shell,
            check=check,
            capture_output=capture_output,
            text=True,
            timeout=timeout
        )
        if capture_output:
            print("STDOUT:\n", result.stdout)
            if result.stderr:
                print("STDERR:\n", result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if capture_output:
            print("STDOUT:\n", e.stdout)
            print("STDERR:\n", e.stderr)
        raise
    except subprocess.TimeoutExpired:
        print(f"Command timed out after {timeout} seconds.")
        raise
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        raise

def check_node_modules():
    """检查node_modules是否存在且有效."""
    node_modules_path = os.path.join(FRONTEND_DIR, "node_modules")
    if not os.path.exists(node_modules_path):
        return False

    # 检查关键包是否存在
    critical_packages = ["react", "react-dom", "react-scripts"]
    for package in critical_packages:
        package_dir = os.path.join(node_modules_path, package)
        if not os.path.exists(package_dir):
            print(f"缺少前端关键包: {package}。需要重新安装依赖。")
            return False
    return True

def install_frontend_dependencies():
    """安装前端Node.js依赖."""
    print("检查并安装前端Node.js依赖...")
    if not os.path.exists(FRONTEND_PACKAGE_JSON):
        print(f"{FRONTEND_PACKAGE_JSON} 不存在，跳过前端依赖安装。")
        return

    if not check_node_modules():
        print("前端依赖不完整或缺失，尝试重新安装...")
        # 删除旧的node_modules和package-lock.json
        node_modules_path = os.path.join(FRONTEND_DIR, "node_modules")
        package_lock_path = os.path.join(FRONTEND_DIR, "package-lock.json")

        if os.path.exists(node_modules_path):
            if platform.system() == "Windows":
                run_command(["rmdir", "/s", "/q", node_modules_path], shell=True)
            else:
                run_command(["rm", "-rf", node_modules_path])
        if os.path.exists(package_lock_path):
            os.remove(package_lock_path)

        try:
            print("运行 npm install...")
            run_command(["npm", "install", "--legacy-peer-deps"], cwd=FRONTEND_DIR, timeout=300)
            print("npm install 成功。")
        except Exception as e:
            print(f"npm install 失败: {e}")
            print("尝试使用 cnpm install...")
            try:
                run_command(["npm", "install", "-g", "cnpm", "--registry=https://registry.npmmirror.com"])
                run_command(["cnpm", "install"], cwd=FRONTEND_DIR, timeout=300)
                print("cnpm install 成功。")
            except Exception as e_cnpm:
                print(f"cnpm install 失败: {e_cnpm}")
                sys.exit(1)
    else:
        print("前端Node.js依赖已存在且有效。")

def check_backend_dependencies():
    """检查关键后端Python依赖是否可导入。"""
    critical_packages = ["flask", "flask_cors", "torch", "torchvision", "numpy", "pandas"]
    missing = [package for package in critical_packages if importlib.util.find_spec(package) is None]
    if missing:
        print(f"缺少后端依赖: {', '.join(missing)}")
        return False
    return True

def install_backend_dependencies():
    """安装后端Python依赖。"""
    if not os.path.exists(BACKEND_REQUIREMENTS):
        print(f"{BACKEND_REQUIREMENTS} 不存在，跳过后端依赖安装。")
        return

    if check_backend_dependencies():
        print("后端Python依赖已存在且有效。")
        return

    print("安装后端Python依赖...")
    run_command([sys.executable, "-m", "pip", "install", "-r", BACKEND_REQUIREMENTS], timeout=600)

def start_backend():
    """启动后端Flask服务器."""
    print("启动后端Flask开发服务器...")
    print("后端API: http://localhost:5000")
    try:
        # 使用Popen以便非阻塞启动
        backend_process = subprocess.Popen(
            [sys.executable, "-m", "flask", "run", "--debug", "--port=5000"],
            cwd=BACKEND_DIR,
            env={**os.environ, "FLASK_APP": "app/__init__.py"}, # 确保FLASK_APP设置正确
            shell=platform.system() == "Windows" # Windows下需要shell=True
        )
        return backend_process
    except Exception as e:
        print(f"启动后端失败: {e}")
        return None

def start_frontend():
    """启动前端React开发服务器."""
    print("启动前端React开发服务器...")
    print("访问地址: http://localhost:3000")
    try:
        # 在 Windows 上使用 npm.cmd start，在其他系统上使用 npm start
        cmd = ["npm", "start"]
        if platform.system() == "Windows":
            # 检查 npm.cmd 是否存在
            cmd = ["npm.cmd", "start"]

        # 使用Popen以便非阻塞启动
        frontend_process = subprocess.Popen(
            cmd,
            cwd=FRONTEND_DIR,
            shell=True # 在 Windows 上启动 .cmd 文件通常需要 shell=True
        )
        return frontend_process
    except Exception as e:
        print(f"启动前端失败: {e}")
        return None

def main():
    """主函数，处理启动逻辑."""
    parser = argparse.ArgumentParser(description="启动联邦学习前后端应用。")
    parser.add_argument("--backend", action="store_true", help="只启动后端。")
    parser.add_argument("--frontend", action="store_true", help="只启动前端。")
    args = parser.parse_args()

    if not (args.backend or args.frontend):
        print("未指定启动模式，将同时启动前端和后端。")
        start_both = True
    else:
        start_both = False

    processes = []

    # 安装依赖
    if args.backend or start_both:
        install_backend_dependencies()
    if args.frontend or start_both:
        install_frontend_dependencies()

    if args.backend or start_both:
        backend_proc = start_backend()
        if backend_proc:
            processes.append(backend_proc)
            time.sleep(5) # 给后端一些时间启动

    if args.frontend or start_both:
        frontend_proc = start_frontend()
        if frontend_proc:
            processes.append(frontend_proc)

    if not processes:
        print("未成功启动任何服务。")
        sys.exit(1)

    print("\n前后端服务已启动。按 Ctrl+C 停止。")
    try:
        while True:
            time.sleep(1)
            # 检查子进程是否仍在运行
            for p in processes:
                if p.poll() is not None: # 进程已终止
                    print(f"一个服务进程已终止，退出。")
                    sys.exit(p.returncode)
    except KeyboardInterrupt:
        print("\n收到停止信号，正在关闭服务...")
    finally:
        for p in processes:
            if p.poll() is None: # 如果进程仍在运行，则终止它
                p.terminate()
                p.wait()
        print("所有服务已关闭。")

if __name__ == "__main__":
    main()

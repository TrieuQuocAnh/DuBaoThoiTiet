import os
import json
import argparse
from pathlib import Path

try:
    import papermill as pm
except ImportError:
    pm = None

def run_notebook(input_nb, output_nb=None, params=None, kernel_name='python3', timeout=600):
    input_path = Path(input_nb).resolve()
    if not input_path.exists():
        print(f"  [!] Không tìm thấy file: {input_path}")
        return False

    if output_nb is None:
        output_path = input_path.with_name(input_path.stem + '_executed' + input_path.suffix)
    else:
        output_path = Path(output_nb).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if pm is None:
        raise ImportError('papermill is not installed. Install via: pip install papermill')

    print(f"\n>>> Đang thực thi: {input_path.name}")
    try:
        pm.execute_notebook(
            input_path=str(input_path),
            output_path=str(output_path),
            parameters=params or {},
            kernel_name=kernel_name,
            progress_bar=True,
            report_mode=False,
            start_timeout=timeout,
            execution_timeout=timeout,
            cwd=str(input_path.parent) # Giữ working directory tại folder notebooks
        )
        print(f"--- Hoàn thành: {output_path.name}")
        return True
    except Exception as e:
        print(f"--- LỖI khi chạy {input_path.name}: {e}")
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chạy hàng loạt Notebooks')
    parser.add_argument('--dir', default='notebooks', help='Thư mục chứa notebooks')
    parser.add_argument('--kernel', default='python3', help='Tên Kernel')
    args = parser.parse_args()

    # 1. Xác định thư mục notebooks
    project_root = Path(__file__).parent.parent # Giả định script nằm trong thư mục 'scripts/'
    nb_dir = project_root / args.dir
    
    if not nb_dir.exists():
        # Nếu không tìm thấy theo cấu trúc dự án, thử tìm ở thư mục hiện tại
        nb_dir = Path.cwd() / args.dir

    if not nb_dir.exists() or not nb_dir.is_dir():
        print(f"Lỗi: Không tìm thấy thư mục '{args.dir}' tại {nb_dir}")
        exit(1)

    # 2. Lấy danh sách các file .ipynb (loại trừ các file đã thực thi)
    notebooks = sorted([f for f in nb_dir.glob("*.ipynb") if "_executed" not in f.name])

    if not notebooks:
        print(f"Không tìm thấy file .ipynb nào trong {nb_dir}")
        exit(0)

    print(f"Tìm thấy {len(notebooks)} notebooks. Bắt đầu chạy lần lượt...")

    # 3. Chạy vòng lặp
    success_count = 0
    for nb in notebooks:
        if run_notebook(nb, kernel_name=args.kernel):
            success_count += 1

    print("\n" + "="*30)
    print(f"TỔNG KẾT: Thành công {success_count}/{len(notebooks)}")
    print("="*30)
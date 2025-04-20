import os
from huggingface_hub import login

# Token API của bạn từ Hugging Face (lấy từ https://huggingface.co/settings/tokens)
HF_API_TOKEN = "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"  # NovaSeele's token main

# Hàm đăng nhập vào Hugging Face
def huggingface_login():
    # Đảm bảo token API được lưu trữ trong môi trường, nếu chưa có
    if not os.getenv("HF_API_TOKEN"):
        os.environ["HF_API_TOKEN"] = HF_API_TOKEN

    # Đăng nhập vào Hugging Face bằng token API
    login(token=os.getenv("HF_API_TOKEN"))
    print("Đăng nhập vào Hugging Face thành công!")

# Gọi hàm đăng nhập khi cần
huggingface_login()

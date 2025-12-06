import gdown
import os


def download_gg_drive_folder(gg_drive_url, output_folder_name):
    folder_id = gg_drive_url.split('/')[-1]
    try:
        gdown.download_folder(
            id=folder_id,
            output=output_folder_name,
            quiet=False,  # Shows progress
            use_cookies=False # Usually not needed for public links
        )

        print(f"\n✅ Successfully downloaded all contents to the local folder: ./{output_folder_name}")

    except Exception as e:
        print(f"\n❌ An error occurred during download: {e}")
        print("Please ensure the folder is genuinely set to 'Public' and not just 'Link sharing on'.")

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
google_drive_urls = [
    (
        'https://drive.google.com/drive/folders/1ACJEoTp-uqfFJ73qS3eUObQh52nGuzCd',
        os.path.join(root_dir, 'BTCV'),
    ),
    (
        'https://drive.google.com/drive/folders/1KQcrci7aKsYZi1hQoZ3T3QUtcy7b--n4',
        os.path.join(root_dir, 'ACDC'),
    )
]

for url, folder_name in google_drive_urls:
    download_gg_drive_folder(url, folder_name)

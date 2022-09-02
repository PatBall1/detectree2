"""Download script for GEE .tif files from Google Drive"""

import argparse
import logging
import pathlib

from src.constants import DATA_PATH
from src.data.gdrive import DriveAPI
from tqdm.autonotebook import tqdm

DEFAULT_SAVE_PATH = DATA_PATH / "gdrive"


def download_files(
        folder_id: str,
        save_path: str,
        recursive: bool = True,
        overwrite: bool = False,
        logger: logging.Logger = logging.getLogger(),
) -> None:
    """
    Download all .tif files in a given directory. Can work recursively.
    Args:
        folder_id (str): gdrive id of the folder to download
        save_path (pathlib.Path): path to save the downloaded data at
        recursive (bool, optional): If True, downloads all subfolders recursively.
            Defaults to True.
        overwrite (bool): Whether to overwrite existing files. Defaults to False.
        logger (logging.Logger): Optional argument to pass a logger to the download
            script for debugging. Defaults to logging.getLogger().
    """

    elements = gdrive.list_files_in_folder(folder_id)
    save_path.mkdir(exist_ok=True)

    progress_bar = tqdm(
        elements,
        position=0,
        leave=True,
    )
    for element in progress_bar:
        file_id = element["id"]
        file_name = element["name"]
        progress_bar.set_description(f"Working on {file_name}")

        # If element is a directory, recursively clone:
        if recursive and gdrive.is_folder(file_id):
            subdir_path = save_path / file_name
            logger.debug("Creating path at %s", subdir_path)

            # Create subfolder and start recursive download
            subdir_path.mkdir(mode=0o777, exist_ok=True)
            download_files(file_id, subdir_path)

        # Download tifs
        elif gdrive.is_tif(file_id):
            file_path = save_path / file_name

            if file_path.exists() and not overwrite:
                logger.info("File %s already exists. Not overwriting.", file_path)
            else:
                logger.debug("Saving file at %s", file_path)
                # Save file
                gdrive.file_download(
                    file_id,
                    save_path=file_path,
                    chunksize=800 * 1024 * 1024,  # 800 MB
                    verbose=False,
                )
                file_path.chmod(0o664)
        # Download tifs
        elif gdrive.is_kml(file_id):
            file_path = save_path / file_name

            if file_path.exists() and not overwrite:
                logger.info("File %s already exists. Not overwriting.", file_path)
            else:
                logger.debug("Saving file at %s", file_path)
                # Save file
                gdrive.file_download(
                    file_id,
                    save_path=file_path,
                    chunksize=800 * 1024 * 1024,  # 800 MB
                    verbose=False,
                )
                file_path.chmod(0o664)

        # Print warnings for other files
        else:
            # logger.warning("Unknown file of type %s", gdrive.get_mime_type(file_id))
            logger.warning("Unknown file of type")
            file_path = save_path / file_name

            if file_path.exists() and not overwrite:
                logger.info("File %s already exists. Not overwriting.", file_path)
            else:
                logger.debug("Saving file at %s", file_path)
                # Save file
                gdrive.file_download(
                    file_id,
                    save_path=file_path,
                    chunksize=800 * 1024 * 1024,  # 800 MB
                    verbose=False,
                )
                file_path.chmod(0o664)


if __name__ == "__main__":

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Google Drive TIF download script")
    parser.add_argument(
        "gdrive_folder_name",
        help="The name of the gdrive folder from which to download the TIF files from.",
        type=str,
    )
    parser.add_argument(
        "save_path",
        help="The folder in which to save the downloaded files.",
        type=str,
        default=DEFAULT_SAVE_PATH,
        nargs="?",  # Argument is optional
    )
    parser.add_argument(
        "-r",
        "--recursive",
        help="If True, copy content of gdrive folder recursively. Defaults to True.",
        type=bool,
        default=True,
        nargs="?",  # Argument is optional
    )
    parser.add_argument(
        "-o",
        "--overwrite",
        help=("If True, existing files are downloaded again and overwritten. "
              "Defaults to False."),
        type=bool,
        default=False,
        nargs="?",  # Argument is optional
    )

    args = parser.parse_args()

    # Connect to google drive and set folders to download and save path
    gdrive = DriveAPI()
    print("Signing in to Google")
    print(f"Signed in as {gdrive.username} ({gdrive.user_email})")
    gdrive_folder_id = gdrive.get_folder_id(args.gdrive_folder_name)
    local_save_path = pathlib.Path(args.save_path)

    # Start download
    print(f"Starting download of {args.gdrive_folder_name}. Saving to {local_save_path}")
    download_files(
        folder_id=gdrive_folder_id,
        save_path=local_save_path,
        recursive=args.recursive,
        overwrite=args.overwrite,
    )

"""Wrapper around Google Drive API (v3) to download files from GDrive"""

import io
import os.path
import pathlib
import pickle
import shutil
from collections import deque
from os import PathLike
from typing import Dict, List, Optional, Union

from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from src.constants import PROJECT_PATH
from src.utils.logging import get_logger
from tqdm.autonotebook import tqdm

logger = get_logger(__file__)
SECRETS_PATH = PROJECT_PATH / "secrets"
DriveFileJson = Dict[str, str]


class DriveAPI:
    """
    Python wrapper around the google drive v3 API.
    Handels OAuth connection, file browsing, meta data retrieval, file download and
    file upload.
    """

    # Define the scopes
    SCOPES = [
        "https://www.googleapis.com/auth/drive",  # For reading and writing
        "https://www.googleapis.com/auth/drive.readonly",  # For reading only (download)
    ]
    # Define GDrive types
    GDRIVE_FOLDER = "application/vnd.google-apps.folder"

    def __init__(
        self,
        credentials_path: Union[str, PathLike] = SECRETS_PATH / "credentials.json",
    ):

        # Variable self.creds will store the user access token.
        # If no valid token found we will create one.
        self.creds = None
        self.credentials_path = pathlib.Path(credentials_path)
        self._user_data = None

        # Authenticate
        self._authenticate()

        # Connect to the API service
        self.service = build("drive", "v3", credentials=self.creds)

    def _authenticate(self) -> None:
        """
        Authenticate user with user token from google OAuth 2.0.
        """
        # The file token.pickle stores the user's access and refresh tokens. It is
        # created automatically when the authorization flow completes for the first
        # time.

        # Check if file token.pickle exists
        if os.path.exists(SECRETS_PATH / "token.pickle"):

            # Read the token from the file and
            # store it in the variable self.creds
            with open(SECRETS_PATH / "token.pickle", "rb") as token:
                self.creds = pickle.load(token)

        # If no valid credentials are available,
        # request the user to log in.
        if not self.creds or not self.creds.valid:

            # If token is expired, it will be refreshed,
            # else, we will request a new one.
            if self.creds and self.creds.expired and self.creds.refresh_token:
                self.creds.refresh(Request())
            else:
                self._perform_oauth()

            # Save the access token in token.pickle
            # file for future usage
            with open(SECRETS_PATH / "token.pickle", "wb") as token:
                pickle.dump(self.creds, token)

    def _perform_oauth(self) -> None:
        """
        Perform google OAuth 2.0 flow to authenticate user.
        """
        flow = InstalledAppFlow.from_client_secrets_file(self.credentials_path, DriveAPI.SCOPES)
        self.creds = flow.run_local_server(port=0)

    @property
    def user_data(self) -> dict:
        """Returns metadata of currently logged in user"""
        if self._user_data is None:
            # fetch user data
            about = self.service.about()
            self._user_data = about.get(fields="user").execute()["user"]
        return self._user_data

    @property
    def user_email(self) -> str:
        """Returns email address of currently logged in user"""
        return self.user_data["emailAddress"]

    @property
    def username(self) -> str:
        """Returns user name of currently logged in user"""
        return self.user_data["displayName"]

    def file_download(
        self,
        file_id: str,
        save_path: str,
        chunksize: int = 200 * 1024 * 1024,
        verbose: bool = False,
    ) -> bool:
        """
        Download file with given `file_id` and save in `save_path`.
        Raises an error if the download fails.
        Args:
            file_id (str): id of the file to download
            save_path (str): path where the file will be saved
            chunksize (int, optional): size of the chunks of data to request with
                each http request. If the download is slow, try increasing the chunksize
                as google limits the number of http requests we can pose per second.
                Defaults to 200*1024*1024 (= 200 MB).
            verbose (bool): Iff true, show download progress for each file. Defaults to
                False.
        Returns:
            bool: True, iff the file was downloaded successfully.
        """
        request = self.service.files().get_media(fileId=file_id)
        file_handle = io.BytesIO()

        # Initialise a downloader object to download the file
        downloader = MediaIoBaseDownload(file_handle, request, chunksize=chunksize)
        done = False

        if verbose:
            print("Starting file download")
        progress_bar = tqdm(total=100, disable=not verbose)
        while not done:
            status, done = downloader.next_chunk()
            if status and verbose:
                progress_bar.update(n=status.progress() * 100)
        progress_bar.close()

        file_handle.seek(0)

        # Write the received data to the file
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file_handle, f)

        if verbose:
            print("File Downloaded")
        # Return True if file Downloaded successfully
        return True

    def get_mimetype(self, file_id: str) -> str:
        """
        Returns mime type of the given file
        Args:
            file_id (str): id of the file
        Returns:
            str: mime type of the given file
        """

        query = self.service.files().get(
            fileId=file_id,
            fields="mimeType",
            supportsAllDrives=True,
        )
        mime_type = query.execute()["mimeType"]

        return mime_type

    def is_mimetype(self, file_id: str, target_mime_type: str) -> bool:
        """
        Check mime type of a given file against target mime type
        Args:
            file_id (str): id of the file
            target_mime_type (str): target mime type to check against
        Returns:
            bool: True, iff the mime type of the given file matches the target mime
                type
        """

        return self.get_mimetype(file_id) == target_mime_type

    def is_folder(self, file_id: str) -> bool:
        """
        Checks if a given file is a gdrive folder
        Args:
            file_id (str): id of the file
        Returns:
            bool: True, iff file is a gdrive folder
        """

        return self.is_mimetype(file_id=file_id, target_mime_type=DriveAPI.GDRIVE_FOLDER)

    def is_tif(self, file_id: str) -> bool:
        """
        Checks if a given file is a .tiff file.
        Args:
            file_id (str): id of the file
        Returns:
            bool: True, iff file is of type .tiff
        """

        return self.is_mimetype(file_id, target_mime_type="image/tiff")

    def is_kml(self, file_id: str) -> bool:
        """
        Checks if a given file is a .kml file.
        Args:
            file_id (str): id of the file
        Returns:
            bool: True, iff file is of type .kml
        """

        return self.is_mimetype(file_id, target_mime_type="application/vnd.google-earth.kml+xml")

    def get_folder(self, folder_name: str, all_drives: bool = True, trashed_ok: bool = False) -> DriveFileJson:
        """
        Return metadata of gdrive folder with the given `folder_name`
        Raises an error if `folder_name` does not identify a unique folder (or does
        not exist).
        Args:
            folder_name (str): The name of the folder for which to obtain metadata'
            all_drives (bool): Whether to search TeamDrives. Defaults to True.
            trashed_ok (bool): Whether to include bin in search. Defaults to False.
        Returns:
            DriveFileJson: The metadata of the requested folder as python dict.
        """

        file_browser = self.service.files()
        file_metadata = {"name": folder_name, "mimeType": self.GDRIVE_FOLDER}
        # Fomulate http request
        query = file_browser.list(
            q=self._metadata_to_query_string(file_metadata=file_metadata, trashed_ok=trashed_ok),
            pageSize=1000,
            supportsAllDrives=all_drives,
            includeItemsFromAllDrives=all_drives,
        )
        # Send http request
        result = query.execute()["files"]

        if len(result) == 0:
            raise UserWarning("No folder with this name exists")
        elif len(result) > 1:
            results = "\n".join([str(elem) for elem in result])
            raise UserWarning(f"Multiple folders with this name exist: \n\n{results}")

        return result[0]

    def get_folder_id(self, folder_name: str, all_drives: bool = True) -> str:
        """
        Return id of a folder with the given name.
        Raises an error if folder does not exist or if multiple folder share the
        same name.
        Args:
            folder_name (str): The folder whose id should be returned
            all_drives (bool): Whether to search TeamDrives. Defaults to True.
        Returns:
            str: id of the folder with the given foldername.
        """

        folder = self.get_folder(folder_name, all_drives=all_drives)

        return folder["id"]

    def get_file_name(self, file_id: str, all_drives: bool = True) -> str:
        """
        Get name of a file by id
        Args:
            file_id (str): The id of the file whose name should be returned
            all_drives (bool): Whether to search TeamDrives. Defaults to True.
        Returns:
            str: The filename
        """

        query = self.service.files().get(fileId=file_id, fields="name", supportsAllDrives=all_drives)
        return query.execute()["name"]

    def list_all_files(self, all_drives=True) -> List[DriveFileJson]:
        """
        Lists all files which are not folders in gdrive
        Returns:
            List[DriveFileJson]: A list of all files in the given gdrive account
        """

        file_browser = self.service.files()
        query = file_browser.list(
            q=f"mimeType!='{self.GDRIVE_FOLDER}'",
            pageSize=1000,
            supportsAllDrives=all_drives,
            includeItemsFromAllDrives=all_drives,
        )
        return query.execute()["files"]

    def list_all_folders(self, all_drives=True) -> List[DriveFileJson]:
        """
        List all folders in gdrive
        Returns:
            List[dict]: List of all folders and their id's
        """

        file_browser = self.service.files()
        query = file_browser.list(
            q=f"mimeType='{self.GDRIVE_FOLDER}'",
            supportsAllDrives=all_drives,
            includeItemsFromAllDrives=all_drives,
        )
        return query.execute()["files"]

    def list_all_drives(self) -> List[DriveFileJson]:
        """
        List all drives
        Returns:
            List[dict]: List of all drives and their id's
        """
        query = self.service.drives().list()
        return query.execute()["drives"]

    def list_files_in_folder(
        self,
        folder_id: str,
        fields: str = "files (id, name)",
        all_drives: bool = True,
        **kwargs,
    ) -> List[DriveFileJson]:
        """
        List all files in a gdrive folder with given `folder_id`.
        Args:
            folder_id (str): The id of the folder
            fields (str, optional): The fields to list. Possible values can be taken
            from the gdrive api v3 documentation. Defaults to "files (id, name)".
        Returns:
            List[dict]: A list of all the files in the given folder.
        """

        file_browser = self.service.files()

        assert self.is_folder(folder_id), "Selected file is not a folder"

        query = file_browser.list(
            q=f"'{folder_id}' in parents",
            fields=fields,
            pageSize=1000,
            supportsAllDrives=all_drives,
            includeItemsFromAllDrives=all_drives,
            **kwargs,
        )
        return query.execute()["files"]

    @staticmethod
    def _metadata_to_query_string(file_metadata: DriveFileJson, trashed_ok: bool = False) -> str:
        """
        Turns file metadata into query string to be used in GDrive API file queries.

        Args:
            file_metadata (DriveFileJson): The metadata to turn into a query
            trashed_ok (bool, optional): Whether to allow documents in trash in query.
                Defaults to False.

        Returns:
            str: The query string to find the file with `file_metadata` on GDrive
        """
        query_str = f"name='{file_metadata['name']}'"
        if "parents" in file_metadata:
            query_str += f" and '{file_metadata['parents'][0]}' in parents"
        if "mimeType" in file_metadata:
            query_str += f" and mimeType='{file_metadata['mimeType']}'"
        query_str += f" and trashed={'true' if trashed_ok else 'false'}"
        return query_str

    def get_file(
        self,
        file_metadata: DriveFileJson,
        trashed_ok: bool = False,
        all_drives: bool = True,
    ) -> List[DriveFileJson]:
        """
        Returns full metadata for all files which match the given `file_metadata`.

        Args:
            file_metadata (DriveFileJson): The file metadata values to use as query
                parameters.
            trashed_ok (bool, optional): Whether to allow documents in trash in query.
                Defaults to False.
            all_drives (bool, optional): Whether to include all drives in the search,
                (i.e. also TeamDrives). Defaults to True.

        Returns:
            List[DriveFileJson]: A list with metadata of all files which match the
                values in `file_metadata`
        """
        query_str = self._metadata_to_query_string(file_metadata, trashed_ok)
        print(query_str)
        return (self.service.files().list(
            q=query_str,
            supportsAllDrives=all_drives,
            includeItemsFromAllDrives=all_drives,
            pageSize=1000,
        ).execute()["files"])

    def exists(
        self,
        file_metadata: DriveFileJson,
        trashed_ok: bool = False,
        all_drives: bool = True,
    ) -> bool:
        """Returns True, iff a file with the given file_metadata exists on GDrive."""
        return len(self.get_file(file_metadata, trashed_ok, all_drives)) > 0

    @staticmethod
    def _add_parent_to_metadata(file_metadata: DriveFileJson, parent: DriveFileJson) -> DriveFileJson:
        """
        Adds parent information to a given metadata template.

        If parent is part of a team drive, team drive information will be passed on
        as well.

        Args:
            file_metadata (DriveFileJson): metadata to modify such that it will include
                parent as its parent
            parent (DriveFileJson): metadata of the parent

        Returns:
            DriveFileJson: The modified file_metadata with `parent` as a parent.
        """
        file_metadata["parents"] = [parent["id"]]
        if "driveId" in parent.keys():
            file_metadata["driveId"] = parent["driveId"]
        if "teamDriveId" in parent.keys():
            file_metadata["teamDriveId"] = parent["teamDriveId"]
        return file_metadata

    def create_folder(
        self,
        folder_name: str,
        parent: Optional[DriveFileJson] = None,
        exists_ok: bool = True,
    ) -> bool:
        """
        Creates a new folder under parent on GDrive.

        Args:
            folder_name (str): Name of the folder to
            parent (Optional[DriveFileJson], optional): The created parent will be a
                subfolder of parent. Defaults to None.
            exists_ok (bool, optional): Whether to ignore existing files.
                If False, existing folder may be overwritten. Defaults to True.

        Returns:
            bool: True iff folder creation succeeded.
        """
        # Write metadata for creating a folder
        file_metadata = {"name": folder_name, "mimeType": self.GDRIVE_FOLDER}
        # Check if folder already exists
        if exists_ok and self.exists(file_metadata):
            print("Folder exists already")
        # Else, create
        else:
            if parent is not None:
                self._add_parent_to_metadata(file_metadata, parent)
            # Execute folder creation request
            self.service.files().create(body=file_metadata, fields="id", supportsAllDrives=True).execute()
        return True

    def upload_file(
        self,
        file_to_upload: pathlib.Path,
        parent: Optional[DriveFileJson] = None,
        exists_ok: bool = True,
        chunksize: int = 5 * 1024 * 1024,
    ) -> bool:
        """
        Uploads the file `file_to_upload` to GDrive.

        Args:
            file_to_upload (pathlib.Path): Path to the file to upload
            parent (Optional[DriveFileJson], optional): GDrive folder under which to
                store the uploaded file. Defaults to None.
            exists_ok (bool, optional): If exists_ok, existing files in with same parent
                 and name will not be overwritten. Defaults to True.
            chunksize (int, optional): The chunksize to use for uploads (default: 5MB).
                Defaults to 5*1024*1024.

        Returns:
            bool: True, iff upload was successful.
        """
        # Define metadata and media to upload
        assert file_to_upload.is_file()
        file_metadata = {"name": file_to_upload.name}
        if parent is not None:
            self._add_parent_to_metadata(file_metadata, parent)
        # Check if file already exists
        if self.exists(file_metadata) and exists_ok:
            print("File exists already.")
            return True
        # If not, upload
        else:
            media = MediaFileUpload(file_to_upload, chunksize=chunksize, resumable=True)
            # Set up http request to upload file
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id",
                supportsAllDrives=True,
            )
            # Upload file in chunks of the given size
            response = None
            progress_bar = tqdm(desc=file_metadata["name"], total=100)
            while response is None:
                status, response = file.next_chunk()
                if status:
                    progress_bar.update(status.progress())
            logger.debug("Upload of %s complete!", file_to_upload.name)
            return True

    def upload_folder(
        self,
        folder_to_upload: pathlib.Path,
        parent: Optional[DriveFileJson] = None,
        chunksize: int = 5 * 1024 * 1024,
    ) -> bool:
        """
        Uploads a folder incl. all subfolders and files to GDrive.

        Note: The folder structure is replicated on GDrive.

        Args:
            folder_to_upload (pathlib.Path): Path to the folder to upload.
            parent (Optional[DriveFileJson], optional): Parent folder under which to put
            the uploaded folder. The uploaded folder will be a subfolder of parent.
                Defaults to None.
            chunksize (int, optional): The chunksize to use for uploads (default: 5 MB).
                Defaults to 5*1024*1024.

        Raises:
            RuntimeError: If a file is neither a directory nor a file

        Returns:
            bool: True, iff the download succeeds
        """

        # Load folder into queue
        assert folder_to_upload.is_dir()
        queue = deque([folder_to_upload])

        # Perform breadth first search traversal until queue is empty
        while len(queue) > 0:
            current_element = queue.popleft()
            logger.debug("Visiting %s", (current_element))

            # Check if current node is a file
            if current_element.is_file():
                parent_folder_name = current_element.absolute().parent.name
                logger.debug("Parent folder: %s", {parent_folder_name})
                parent = self.get_folder(folder_name=parent_folder_name)
                self.upload_file(current_element, parent=parent, exists_ok=True, chunksize=chunksize)
            # Else it is current node is a folder and we have to look at all children
            elif current_element.is_dir():
                # Note: In first iteration, current_element will be folder_to_upload
                #  and thus parent is the parent given in the function signature in
                #  that case.
                if current_element != folder_to_upload:
                    parent_folder_name = current_element.absolute().parent.name
                    parent = self.get_folder(folder_name=parent_folder_name)
                self.create_folder(current_element.name, parent=parent, exists_ok=True)
                folder_contents = list(current_element.glob("*"))
                queue.extendleft(folder_contents)
            else:
                raise RuntimeError

        return True

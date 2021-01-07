from . import flowcam_loader
import itertools

class data_management_engine():
    def __init__(self, path, _path_list = None, query_string = "all"):
        """
        the initiator of a data management engine, create a list of flowcam
        loader for all the valid folders inside, note the detection of flowcam loader
        is by the existence of "data_export.csv" file
        args:
            path: str, the root path, dme will search through all the folders recursively below
            _path_list: list of string, private varible for "query" method, if not None,
                        the query will be in the list given
            query_string: str/iteratbles, the query condition,
                          dme will check the path of all the folders searched
                          only the path with {query_string} will be included
        """
        self._root_path = path
        # when using a public interface
        if _path_list == None:
            # if querying all the strings
            if query_string == "all":
                self._query_string = ["all"]
                # search for all the data_exprt.csv and get their folder name
                self._path_list = [os.path.dirname(i) for i in glob2.iglob(os.path.join(path,"**/data_export.csv"))]
                print(f"{len(self._path_list)} folders are loaded from {path}")
            # if given some queries
            else:
                # convert query(s) into correct format
                if isinstance(query_string, str):
                   self._query_string = [query_string]
                else:
                   self._query_string = [i for i in query_string]
                # query all the folders with given queries, all the conditions must be met
                self._path_list = [os.path.dirname(i) for i in glob2.iglob(os.path.join(path,"**/data_export.csv"))\
                                  if all(subqueries in os.path.dirname(i) for subqueries in self._query_string)]
                print(f"{len(self._path_list)} folders are loaded by query {self._query_string} from {path}")
        # when using a private interface
        else:
            # inherite the query strings specified
            self._query_string = query_string
            # search through the _path_list range with conditions
            self._path_list = [i for i in  _path_list\
                              if all(subqueries in i for subqueries in self._query_string)]
            print(f"{len(self._path_list)} folders are loaded by query {self._query_string} from {path}")
        # load the actual loaders
        self._loaders = [flowcam_loader(path = paths) for paths in self.path_list]


    def __len__(self):
        """
        return the number of loaders in the management engine object

        Return:
            int, the number of loaders inside the engine
        """
        return len(self.path_list)

    def __getitem__(self, i):
        """
        return the specific loader inside the engine

        Return:
            flowcam_loader object, the specific loader in the engine
        """
        return self._path_list[i]

    def query(self, query):
        """
        return a data management loader object searched by search_string

        Return:
            data_management_engine object, a object with a subset of loaders
        """
        return data_management_engine(
            path = self._root_path, # inherite the parent folder
            _path_list = self._path_list, # inherite the parent path_list
            # the query condition will skip all the "all"s and add addtional query
            query_string = [i for i in self._query_string if i != "all"] + [query]
        )

    @property
    def query_from(self):
        return self._query_string

    @property
    def root_path(self):
        return self._root_path

    @property
    def path_list(self):
        return self._path_list

    def load_features(self):
        """
        have all the loaders load their features one by one,
        for memory consideration, the features will not be saved inside the object, which might be extremely large
        Return:
          pandas.DataFrame, the feature of all the features under each folder
        """
        return pd.concat(loader.load_features_temp() for loader in self._loaders)

    def load_images(self):
        """
        have all the loaders load their images one by one,
        for memory consideration, the features will not be saved inside the object

        Return
          itertools.chain generator object, I don't want to save all of them into memory
        """
        return itertools.chain(loader.load_images_temp() for loader in self._loaders)

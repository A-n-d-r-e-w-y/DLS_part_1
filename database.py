from __init__ import DEFAULT_PARAMS


class DataStore(object):
    def __init__(self):
        self.chatId_to_files = dict()
        self.chatId_to_state = dict()
        self.chatId_to_params = dict()

    def update_files(self, chat_id, file_id, file_type):
        if self.chatId_to_files.get(chat_id, -1) == -1:
            self.chatId_to_files[chat_id] = {
                "photo": None,
                "style_1": None,
                "style_2": None,
            }
        self.chatId_to_files[chat_id][file_type] = file_id

    def get_files(self, chat_id: int):
        return self.chatId_to_files.get(chat_id, -1)

    def update_state(self, chat_id: int, state: str):
        self.chatId_to_state[chat_id] = state

    def get_state(self, chat_id: int):
        return self.chatId_to_state.get(chat_id, "null_state")

    def update_params(self, chat_id: int, params: dict):
        if self.chatId_to_params.get(chat_id, -1) == -1:
            self.chatId_to_params[chat_id] = DEFAULT_PARAMS
        for param_name, param_value in params.items():
            self.chatId_to_params[chat_id][param_name] = param_value

    def get_params(self, chat_id: int):
        if self.chatId_to_params.get(chat_id, -1) == -1:
            self.chatId_to_params[chat_id] = DEFAULT_PARAMS
        return self.chatId_to_params[chat_id]

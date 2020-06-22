import logging

from formatter.qa.Bert import BertQA, BertXQA
from formatter.qa.Char import CharQA
from formatter.qa.Comatch import ComatchingFormatter
from formatter.qa.HAF import HAFQA
from formatter.qa.Word import WordQA

logger = logging.getLogger(__name__)

formatter_list = {
    "BertQA": BertXQA,
    "CharQA": CharQA,
    "WordQA": WordQA,
    "ComatchQA": ComatchingFormatter,
    "HAF": HAFQA
}


def init_formatter(config, mode, *args, **params):
    temp_mode = mode
    if mode != "train":
        try:
            config.get("data", "%s_formatter_type" % temp_mode)
        except Exception as e:
            logger.warning(
                "[reader] %s_formatter_type has not been defined in config file, use [dataset] train_formatter_type instead." % temp_mode)
            temp_mode = "train"
    which = config.get("data", "%s_formatter_type" % temp_mode)

    if which in formatter_list:
        formatter = formatter_list[which](config, mode, *args, **params)

        return formatter
    else:
        logger.error("There is no formatter called %s, check your config." % which)
        raise NotImplementedError

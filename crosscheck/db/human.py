from typing import List, Dict
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer,
    String, Boolean, LargeBinary, DateTime
)
from crosscheck import definitions, version
import datetime
import pathlib


masks = list(reversed(['UP', 'DOWN', 'LEFT', 'RIGHT', 'A', 'B', 'C']))

def encode(controls: List[Dict]) -> bytes:
    encoded = []
    for control in controls:
        value = 0
        for shift, label in enumerate(masks):
            if control.get(label):
                mask = (1 << shift)
                value |= mask
        encoded.append(value)
    return bytes(encoded)

def decode(encoded: bytes) -> List[Dict]:
    controls = []
    for value in encoded:
        control = {}
        for shift, label in enumerate(masks):
            mask = (1 << shift)
            control[label] = (mask & value) > 0
        controls.append(control)
    return controls


def add_play(scenario: str, success: bool, controls: List[Dict], db_filename: pathlib.Path = None):

    if db_filename is None:
        db_filename = definitions.DB_FILE
    engine = create_engine(f'sqlite:///{str(db_filename.resolve())}', echo=True)
    meta = MetaData()

    human_games = Table(
        'human_games', meta,
        Column('id', Integer, primary_key=True),
        Column('scenario', String),
        Column('success', Boolean),
        Column('controls', LargeBinary),
        Column('version', String),
        Column('datetime', DateTime)
    )
    meta.create_all(engine)

    controls_bytes = encode(controls)
    ins = human_games.insert()
    ins = human_games.insert().values(
        scenario=scenario,
        success=success,
        controls=controls_bytes,
        version=version.__version__,
        datetime=datetime.datetime.now())

    conn = engine.connect()
    _result = conn.execute(ins)
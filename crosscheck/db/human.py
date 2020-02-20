from typing import List, Dict
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Boolean, LargeBinary



masks = ['UP', 'DOWN', 'Left', 'Right', 'A', 'B', 'C']

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
            control[label] = mask & value
        controls.append(control)
    return controls


def add_play(scenario: str, success: bool, controls: List[Dict]):
    engine = create_engine('sqlite:///college.db', echo=True)
    meta = MetaData()

    students = Table(
        'human', meta,
        Column('id', Integer, primary_key=True),
        Column('scenario', String),
        Column('success', Boolean),
        Column('controls', LargeBinary),
    )

    ins = students.insert()
    ins = students.insert().values(scenario=scenario, success=success, controls=controls_bytes)
    conn = engine.connect()
    result = conn.execute(ins)
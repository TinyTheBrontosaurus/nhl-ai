from typing import List, Dict
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, backref
from sqlalchemy.orm import sessionmaker
from sqlalchemy import (
    create_engine, MetaData, Table, Column, Integer,
    String, Boolean, LargeBinary, DateTime, ForeignKey
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


Base = declarative_base()
class HumanRound(Base):
    __tablename__ = "human_round"
    human_game_id = Column(Integer, primary_key=True)
    scenario_filename = Column(String, ForeignKey('scenario.filename'))
    version_git_version = Column(String, ForeignKey('version.git_version'))
    game_time_start = Column(DateTime, ForeignKey('game_time.start'))
    version = relationship('Version', backref='human_round')
    success = Column(Boolean)
    controls = Column(LargeBinary)
    roundtime = Column(DateTime)

class GameTime(Base):
    __tablename__ = "game_time"
    start = Column(DateTime, primary_key=True)

class Version(Base):
    __tablename__ = "version"
    git_version = Column(String, primary_key=True)

class Scenario(Base):
    __tablename__ = "scenario"
    filename = Column(String, primary_key=True)



def add_round(scenario: pathlib.Path, success: bool, controls: List[Dict], gametime: datetime.datetime, db_filename: pathlib.Path = None):

    if db_filename is None:
        db_filename = definitions.DB_FILE
    engine = create_engine(f'sqlite:///{str(db_filename.resolve())}')
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    scenario_rel = str(scenario.relative_to(definitions.SAVE_STATE_FOLDER))
    sr = get_or_create(session, Scenario, filename=scenario_rel)
    vr = get_or_create(session, Version, git_version=version.__version__)
    gr = get_or_create(session, GameTime, start=gametime)

    controls_bytes = encode(controls)
    new_round = HumanRound(
        scenario_filename=sr.filename,
        version_git_version=vr.git_version,
        game_time_start=gr.start,
        success=success,
        controls=controls_bytes,
        roundtime=datetime.datetime.now()
    )

    session.add(new_round)
    session.commit()


def read_plays(db_filename: pathlib.Path = None):

    if db_filename is None:
        db_filename = definitions.DB_FILE

    engine = create_engine(f'sqlite:///{str(db_filename.resolve())}')
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


# https://stackoverflow.com/a/6078058/5360912
def get_or_create(session, model, **kwargs):
    instance = session.query(model).filter_by(**kwargs).first()
    if instance:
        return instance
    else:
        instance = model(**kwargs)
        session.add(instance)
        session.commit()
        return instance

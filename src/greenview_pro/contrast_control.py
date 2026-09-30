"""Two-ended display contrast control."""

from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QWidget


class ContrastRange(QWidget):
    changed = Signal(int, int)
    MIN_GAP = 2

    def __init__(self, low, high, parent=None, *, minimum=0, maximum=100):
        super().__init__(parent)
        if maximum <= minimum:
            raise ValueError("Contrast range requires maximum above minimum")
        self._minimum = minimum
        self._maximum = maximum
        self._low = minimum
        self._high = maximum
        self._drag_target = None
        self._drag_start_x = 0
        self._drag_start_values = (minimum, maximum)
        self.setMinimumWidth(170)
        self.setFixedHeight(46)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setAccessibleName("Contrast range")
        self.setToolTip(
            "Drag either grip or the selected band. Arrow keys: left/right adjust "
            "the minimum, up/down adjust the maximum; Shift+left/right moves both."
        )
        self.set_values(low, high)

    def sizeHint(self):
        return QSize(240, 46)

    def values(self):
        return self._low, self._high

    def set_values(self, low, high):
        if not (
            self._minimum <= low <= high - self.MIN_GAP
            and high <= self._maximum
        ):
            raise ValueError("Contrast bounds require a gap of at least 2 steps")
        if (low, high) != (self._low, self._high):
            self._low, self._high = low, high
            self.update()
            self.changed.emit(low, high)

    def _track_rect(self):
        return QRect(10, 16, self.width() - 20, 22)

    def _value_x(self, value):
        track = self._track_rect()
        return track.left() + round(
            (value - self._minimum) * track.width()
            / (self._maximum - self._minimum)
        )

    def _x_value(self, x):
        track = self._track_rect()
        return max(
            self._minimum,
            min(
                self._maximum,
                self._minimum + round(
                    (x - track.left()) * (self._maximum - self._minimum)
                    / track.width()
                ),
            ),
        )

    def _target_at(self, x):
        low_x, high_x = self._value_x(self._low), self._value_x(self._high)
        if min(abs(x - low_x), abs(x - high_x)) <= 9:
            return "low" if abs(x - low_x) <= abs(x - high_x) else "high"
        if low_x < x < high_x:
            return "range"
        return "low" if x < low_x else "high"

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        track = self._track_rect()
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor("#D9E2E3"))
        painter.drawRoundedRect(track, 5, 5)
        left, right = self._value_x(self._low), self._value_x(self._high)
        painter.setBrush(QColor("#008A96"))
        painter.drawRoundedRect(QRect(left, track.top(), right - left, track.height()), 4, 4)
        painter.setPen(QPen(QColor("#203A40"), 1))
        painter.setBrush(QColor("#FFFFFF"))
        for x in (left, right):
            painter.drawRoundedRect(
                QRect(x - 5, track.top() - 3, 10, track.height() + 6), 3, 3
            )
            painter.drawLine(x, track.top() + 5, x, track.bottom() - 5)
        if self.hasFocus():
            painter.setPen(QPen(QColor("#008A96"), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -2, -2), 5, 5)

    def keyPressEvent(self, event):
        low, high = self.values()
        if event.modifiers() & Qt.ShiftModifier and event.key() in (
            Qt.Key_Left, Qt.Key_Right
        ):
            delta = -1 if event.key() == Qt.Key_Left else 1
            delta = max(self._minimum - low, min(delta, self._maximum - high))
            self.set_values(low + delta, high + delta)
        elif event.key() == Qt.Key_Left:
            self.set_values(max(self._minimum, low - 1), high)
        elif event.key() == Qt.Key_Right:
            self.set_values(min(low + 1, high - self.MIN_GAP), high)
        elif event.key() == Qt.Key_Up:
            self.set_values(low, min(self._maximum, high + 1))
        elif event.key() == Qt.Key_Down:
            self.set_values(low, max(high - 1, low + self.MIN_GAP))
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            super().mousePressEvent(event)
            return
        x = event.position().x()
        self._drag_target = self._target_at(x)
        self._drag_start_x = x
        self._drag_start_values = self.values()
        if self._drag_target == "range":
            self.setCursor(Qt.ClosedHandCursor)
        else:
            self.setCursor(Qt.SizeHorCursor)
            self._move_to(x)
        event.accept()

    def _move_to(self, x):
        low, high = self._drag_start_values
        if self._drag_target == "low":
            self.set_values(min(self._x_value(x), high - self.MIN_GAP), high)
        elif self._drag_target == "high":
            self.set_values(low, max(self._x_value(x), low + self.MIN_GAP))
        elif self._drag_target == "range":
            track_width = self._track_rect().width()
            delta = round(
                (x - self._drag_start_x) * (self._maximum - self._minimum)
                / track_width
            )
            delta = max(self._minimum - low, min(delta, self._maximum - high))
            self.set_values(low + delta, high + delta)

    def mouseMoveEvent(self, event):
        if self._drag_target is not None:
            self._move_to(event.position().x())
        else:
            self.setCursor(
                Qt.OpenHandCursor
                if self._target_at(event.position().x()) == "range"
                else Qt.SizeHorCursor
            )
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self._drag_target is not None:
            self._move_to(event.position().x())
            self._drag_target = None
            self.unsetCursor()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

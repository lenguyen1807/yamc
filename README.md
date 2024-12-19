> NOTE: This project is generate with cmake-init, so writing a building instruction is a waste of time (actually I don't know how to make a build instruction).

**TL'DR**: This is the rewrite branch of my old source (few months ago), the old version took about 37 minutes to finish training and testing when the new version only took 9 minutes (insane improvement).

---

<img src="result_old.png">
The old version

---

<img src="result_new.png">
The new version but without optimization flag (took about 72 minutes LOL).

---

<img src="result_Ofast.png">
The new version but using `-Ofast` optimization flag.

---

<img src="result_O2.png">
The new version with `-O2` optimization flag.

---

Seeing more on my blog: https://lenguyen.vercel.app/projects/cpp-nn
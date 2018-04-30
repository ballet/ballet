class PullRequestBuildDiffer:
    def __init__(self, pr_num, repo):
        self.pr_num = pr_num
        self.repo = repo
        self.check_environment()

    def check_environment(self):
        raise NotImplementedError

    def get_diff_str(self):
        raise NotImplementedError

    def diff(self):
        diff_str = self.get_diff_str()
        diffs = get_diffs_by_diff_str(self.repo, diff_str)
        return diffs


def get_file_changes_by_revision(repo, from_revision, to_revision):
    '''Get file changes between two revisions

    For details on specifying revisions, see

        git help revisions
    '''
    diff_str = '{from_revision}..{to_revision}'.format(
        from_revision=from_revision, to_revision=to_revision)
    return get_diffs_by_diff_str(repo, diff_str)


def get_diffs_by_diff_str(repo, diff_str):
    a, b = diff_str.split('..')
    a_obj = repo.rev_parse(a)
    b_obj = repo.rev_parse(b)
    diffs = a_obj.diff(b_obj)
    return diffs


# deprecated for now
class PullRequestInfo:
    def __init__(self, pr_num):
        self.pr_num = pr_num

    def _format(self, str):
        return str.format(pr_num=self.pr_num)

    @property
    def local_ref_name(self):
        '''Shorthand name of local ref, e.g. 'pull/1' '''
        return self._format('pull/{pr_num}')

    @property
    def local_rev_name(self):
        '''Full name of revision, e.g. 'refs/heads/pull/1' '''
        return self._format('refs/heads/pull/{pr_num}')

    @property
    def remote_ref_name(self):
        '''Full name of remote ref (as on GitHub), e.g. 'refs/pull/1/head' '''
        return self._format('refs/pull/{pr_num}/head')


class HeadInfo:
    def __init__(self, repo):
        self.head = repo.head

    @property
    def path(self):
        return self.head.ref.path
